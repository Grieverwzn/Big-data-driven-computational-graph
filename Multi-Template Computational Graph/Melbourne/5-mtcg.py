"""Step 5: train and evaluate the Multi-Template Computational Graph (MTCG) on the Melbourne days.

Purpose / paper sections
------------------------
Trains the MTCG of paper Section 4 with the block-wise procedure of Section 6 on the Melbourne data and evaluates it
(paper Sections 7.3.2 setup, 7.3.4 block-wise training and Fig. 22, 7.3.5 Table 12 and Fig. 23, 7.3.6 Table 13 and
Figs. 24-26; the tables and figures themselves are made by steps 6-7).

What
----
The model of the paper with S templates (S = 1: Template 1; S = 2: T1 + T2; S = 3: T1 + T2 + T3), trained on the 80
training days and evaluated on the 20 test days of step 3 (time window 7:00-9:45, T = 12 steps of 15 min).
For every template s (one computational graph per template; code variable -> paper symbol):
  - demand layer (Sec. 4.5): a DNN H^(0)_s maps the day's period OD demand q_bar^m (the only demand input; sum over
    the 12 steps) to the shares p_{s,t} of the steps (Eq. 35, softmax over t); the step-t demand q_{s,t} is this share
    times the period demand times the factor alpha_{s,t} = 1 + H^(1)_{s,t}(v_{s,t-1}, tau_{s,t-1}) (Eq. 34) from two
    DNNs fed with the previous step's link flows and OD travel times (Eq. 33); alpha in (0.15, 2), beta_{s,t} = 0
    ("share-based form", Sec. 7.3.2);
  - route-choice layer: Logit split (Eq. 21) of each OD pair's demand over its K = 2 paths with one parameter
    theta_{s,t} per template and step (bounded in [0.1, 1.5] per minute, start 1.0; VOT = 60 theta AUD/h); path times
    pi_{s,t} from the link times of the previous step (Eq. 22; free-flow at 7:00);
  - link layer: link flows v_{s,t} = path flows x path-link matrix delta_s; link times t_a(v) by BPR with the hourly
    flow rate 4 v.
Fusion (Sec. 4.4): an attention layer (query = embedded period demand, keys = embedded link flows of each template and
step; Eqs. 30-31) gives the weights lambda_{s,t}^m that mix the templates' link flows (Eq. 24) and demand (Eq. 25).
Loss terms (Sec. 4.6, Eq. 41; code name -> paper): L1 = L_V, MSE of the link flows on the 403 sensors used for
training (45 sensors withheld for validation; Eq. 36), weight mu_1 = 1; L2 = L_Q, MSE of the period demand of each
template (Eq. 37), mu_2 = 1; L3 = L_D, Beckmann/VDF term (Eq. 38), mu_3 = 0.001; L5a = L_{Q,T}, MSE of each template's
per-step demand against the day's per-step sample demand (Eq. 40), mu_4 = 2000.
Block-wise training (Section 6): rounds of Step 1 demand block 'D' (mu_2 L_Q + mu_4 L_{Q,T}), Step 2 route-choice
block 'TH' (theta; mu_1 L_V + mu_3 L_D) and Step 3 attention block 'A' (mu_1 L_V + mu_3 L_D; skipped for S = 1); up to
3 rounds with at most 20/15/15, 8/10/8, 5/8/6 epochs; a block ends early (after at least 6 epochs) when its own measure
changes less than 1 % (weights: 0.02) between the mean of its last 3 and the 3 epochs before; the rounds end early when
theta, link loss and weights all settle; then Step 4 fine-tuning 'J': 5 epochs with all blocks and halved learning
rates (all four terms). Adam (betas 0.5, 0.999), lr 0.0005 (theta 0.05), batches of 4 days, gradient clipping at 1,
at most 100 epochs; seeds 0.
This is the code of the experiment notebook (3-MTCG-r2v5_S{S}_alt_executed.ipynb) as a script; computations and
settings are unchanged. Additions: CPU thread count, and a checkpoint at the start of every block so that a run
killed by the operating system continues from that block with the same result (delete the checkpoint to restart).

Usage
-----
    python 5-mtcg.py --S 1        (about 10 min, peak memory about 4 GB)
    python 5-mtcg.py --S 2        (about 25 min, about 6 GB)
    python 5-mtcg.py --S 3        (about 45 min, about 6.5 GB)
    options: --threads 8 (torch CPU threads), --epochs 100, --out results/mtcg_S<S>,
             --ckpt-dir <folder> (default: <out>/checkpoint; use a folder outside cloud-synced drives), --min-free-gb 0

Inputs
------
    data/days100_logit_start7_cap4_k2/ (step 3), data/data_new_{1,2,3}/ (steps 1-2)
Outputs (results/mtcg_S<S>/, same files as the experiment)
-------
    error_table.csv (RMSE / MAE / MAPE: sensors used, withheld sensors, all sensors, OD demand; 20 test days),
    error_by_day.csv, theta_st.csv, att.csv, training_history.csv, run_info.json, predictions.npz, path_state.npz,
    path_order_T*.csv, attention_by_day.csv, attention_by_day_train.csv, demand_fit_by_day.csv,
    template_demand_test.npz, days_train.csv, days_test.csv, estimation2.csv, loss_train.xlsx, loss_test.xlsx,
    progress.log

Authors:
    Xin (Bruce) Wu, Department of Civil and Environmental Engineering, Villanova University, PA, USA
    Feng Shao, School of Mathematics, China University of Mining and Technology, China

Contact: xwu03@villanova.edu (Villanova University), xinwu8592@gmail.com (personal)

MIT License
Copyright (c) 2026 Xin (Bruce) Wu, Feng Shao
"""
import argparse
import os
import sys
import re
import math
import json
import time
import random
import platform
import numpy as np
import pandas as pd
from pandas import DataFrame
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import mtcg_common as C

ap = argparse.ArgumentParser()
ap.add_argument('--S', type=int, required=True, choices=[1, 2, 3])
ap.add_argument('--threads', type=int, default=8)
ap.add_argument('--epochs', type=int, default=100)
ap.add_argument('--out', default=None)
ap.add_argument('--ckpt-dir', default=None)
ap.add_argument('--min-free-gb', type=float, default=0.0, help='wait (up to 3 h) until this much memory is available')
args = ap.parse_args()

random.seed(0); np.random.seed(0); torch.manual_seed(0)       # same seeds and order of random draws as the notebook
torch.set_num_threads(args.threads)
S = args.S
OUT = args.out or os.path.join(C.RES, f'mtcg_S{S}')
os.makedirs(OUT, exist_ok=True)
CKPT_DIR = args.ckpt_dir or os.path.join(OUT, 'checkpoint')
CKPT = os.path.join(CKPT_DIR, f'mtcg_S{S}_block_start.pt')
PROGRESS = os.path.join(OUT, 'progress.log')
C.need(os.path.join(C.DAYS, 'train', 'days.npz'), '3-data_generation.py')
if args.min_free_gb > 0 and not C.wait_for_memory(args.min_free_gb):
    sys.exit('not enough free memory after 3 hours')

# ---------------------------------------------------------------------------------------------------------------
# Network and path sets (all three templates are read, as in the notebook; the model uses the first S)
# ---------------------------------------------------------------------------------------------------------------
net = C.load_model_network()
num_od, num_link, k = net['num_od'], net['num_link'], net['k']
free_flow_tt, link_capacity = net['free_flow_tt'], net['link_capacity']
LP0, LPT0, spmm = net['LP0'], net['LPT0'], net['spmm']
path_last1, path_last2, path_last3 = net['path_last']

# ---------------------------------------------------------------------------------------------------------------
# Data and model options (final settings of the experiment)
# ---------------------------------------------------------------------------------------------------------------
DAYS = C.DAYS_NAME
DAY_DIR = C.DAYS
W0, T = C.W0, C.T                  # window start slot (7:00) and number of time steps
BATCH = 4                          # days per training batch
ANCHOR, ANCHOR_SHARE, W_ANCHOR = False, 1.0, None   # template-demand anchor: not used (kept for run_info)
THETA_LO, THETA_HI = 0.1, 1.5      # bounds of theta_{s,t} (1/minute): theta = LO + (HI - LO) sigmoid(u)
THETA_INIT = 1.0                   # initial theta_{s,t}
SHARE_BASE = True                  # step demand = period demand x share (first DNN) x step factor
LR_OTHER = 0.0005                  # learning rate of all parameters except theta (theta: 0.05)
W2_PERIOD = 1.0                    # weight of the period-demand loss L2
DEMAND_T_W = 2000.0                # weight of the per-step OD loss L5a
DEMAND_T_SHARE = 1.0               # per-step OD loss on (kept for run_info)
TRUE_DEMAND = False                # diagnostic switch of the experiment: not used
FLOW_TO_HOURLY = 4                 # link capacity is per hour, flows per 15 min: v/c = 4 v / C
STEP_LO, STEP_HI = 0.15, 2.0       # range of the step factor
STOP_INFO = {'reason': 'max epochs', 'epochs_run': 0}
RUN_TAG = (f'_th{THETA_LO:g}-{THETA_HI:g}_step{STEP_LO:g}-{STEP_HI:g}_cap{FLOW_TO_HOURLY:g}_demtw{DEMAND_T_W:g}'
           f'_w2{W2_PERIOD:g}_lr{LR_OTHER:g}_share_k{k}')
Q_TEMPLATE = torch.from_numpy(np.stack([pd.read_csv(os.path.join(C.template_dir(s), 'demand_6-10.csv')).values.T[W0:W0 + T]
                                        for s in (1, 2, 3)]).astype(np.float32))   # template reference demand (output only)


def theta_of(u):
    return THETA_LO + (THETA_HI - THETA_LO) * torch.sigmoid(u)


def theta_init_raw(theta0):
    p = (theta0 - THETA_LO) / (THETA_HI - THETA_LO)
    return math.log(p / (1 - p))


def _load(split):
    z = np.load(os.path.join(DAY_DIR, split, 'days.npz'))
    info = pd.read_csv(os.path.join(DAY_DIR, split, 'days.csv'))
    dem = torch.from_numpy(z['demand'][:, W0:W0 + T, :])       # (N, T, num_od)
    lf = torch.from_numpy(z['link_flow'][:, W0:W0 + T, :])     # (N, T, num_sensor)
    assert dem.shape[2] == num_od
    return dem, lf, info, z['sensor_ids']


dem_tr, lf_tr, days_train, _sid = _load('train')
dem_te, lf_te, days_test, _sid_te = _load('test')
flow = pd.read_csv(os.path.join(C.template_dir(1), 'link_flow.csv'), nrows=1)     # sensor column order (link ids)
assert (flow.columns.values.astype(float).astype(int) == _sid).all() and (_sid == _sid_te).all()
train_input = dem_tr.sum(1)                                  # (N, num_od) period demand of each day = attention query
test_input = dem_te.sum(1)
train_output = lf_tr.transpose(0, 1).contiguous()            # (T, N, num_sensor)
test_output = lf_te.transpose(0, 1).contiguous()
demand_train_reference = dem_tr.transpose(0, 1).contiguous() # (T, N, num_od) per-step sample demand
demand_test_reference = dem_te.transpose(0, 1).contiguous()
demand_reference = demand_test_reference
num_train = train_input.shape[0]

_PS_PROC = psutil.Process(); _PS_PROC.cpu_percent(); psutil.cpu_percent()
_MAPE_TE = {}


def _mape_now(pred):
    """Test-day link MAPE (|o - p| / o, terms >= 100 dropped), for the progress line."""
    P_ = pred[:, :, observation_link_number]
    for n_, ix_ in (('sens', observed_link_idx), ('nosens', unobserved_link_idx), ('all', slice(None))):
        o_, p_ = test_output[:, :, ix_], P_[:, :, ix_]
        e_ = (o_ - p_).abs() / o_
        _MAPE_TE[n_] = float(e_[e_ < 100].mean())


def _progress(row):
    th_ = np.array([v for k_, v in row.items() if k_.startswith('theta_')])
    n_bd_ = int(((th_ < THETA_LO * 1.1) | (th_ > THETA_HI * 0.95)).sum())
    ep_, el_ = row['epoch'] + 1, row['elapsed_s']
    left_ = el_ / ep_ * (n_epochs - ep_)
    line_ = (f"{time.strftime('%H:%M:%S')} S={S} epoch {ep_}/{n_epochs} updates {row.get('n_updates', '')} | "
             f"elapsed {el_/60:.1f} min ({el_/ep_:.0f} s/epoch) left {left_/60:.1f} min (at most) | "
             f"loss train total {row['train_total']:.0f} link {row['train_link']:.0f} demand {row['train_demand']:.1f} "
             f"demand_t {row.get('train_demand_t', 0):.3f} | test link {row['test_link']:.0f} MAPE sensors "
             f"{_MAPE_TE.get('sens', float('nan')):.1%} no-sensor {_MAPE_TE.get('nosens', float('nan')):.1%} "
             f"all {_MAPE_TE.get('all', float('nan')):.1%} | theta min/median/max {th_.min():.3f}/{np.median(th_):.3f}/"
             f"{th_.max():.3f} at bound {n_bd_}/{th_.size} | mem process {_PS_PROC.memory_info().rss/1e9:.2f} GB "
             f"free {psutil.virtual_memory().available/1e9:.2f} GB")
    try:
        with open(PROGRESS, 'a') as f_:
            f_.write(line_ + chr(10))
    except OSError:
        pass


print('train days:', days_train.type.value_counts().to_dict(), ' test days:', days_test.type.value_counts().to_dict())
print(train_input.shape, train_output.shape, test_input.shape, test_output.shape)


# ---------------------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------------------
def BPR(link_flow, free_flow_tt, link_capacity):
    """Link travel time t_a(v) (BPR, alpha 0.15, beta 4); link_flow per 15 min, capacity per hour."""
    return free_flow_tt * (1 + 0.15 * (FLOW_TO_HOURLY * link_flow / link_capacity) ** 4)


def assignment(demand, theta, path_time, LP):
    """One single-template CG step: Logit split (paper Eq. 21) of the OD demand q_{s,t} over the K paths with
    theta_{s,t} (shared by all OD pairs) and path times pi_{s,t} (Eq. 22); link flows v_{s,t} = f delta_s^T; link times."""
    batch_size = demand.shape[0]
    if path_time.dim() == 1:
        path_time = path_time.repeat(batch_size, 1)
    demand_expand = demand.reshape(-1, 1).repeat(1, k).reshape(batch_size, -1)   # batch x num_path
    utility = (-theta * path_time).reshape(batch_size, -1, k)                      # batch x num_od x k
    P0 = torch.softmax(utility, dim=2)                                             # route shares (stable softmax)
    P = P0.view(batch_size, -1)
    path_flow = demand_expand * P
    link_flow = spmm(LP, path_flow.T).T                                            # = path_flow @ LP.T
    link_time = BPR(link_flow, free_flow_tt, link_capacity)
    return link_flow, link_time


def Initialization(parameters):
    for para in parameters:
        nn.init.kaiming_normal_(para)


class Attention(nn.Module):
    """Attention fusion (paper Sec. 4.4): query = embedded period demand (zeta^Q U^Q), keys = embedded link flows of
    each template and step (v_{s,t} U^K); scores e_{s,t}^m (Eq. 30), weights lambda_{s,t}^m = softmax over s (Eq. 31);
    values = the templates' link flows, so the output is the fused flow sum_s lambda_{s,t}^m v_{s,t} (Eq. 24)."""

    def __init__(self, num_od, num_link, hidden_size):
        super(Attention, self).__init__()
        self.weights_demand = Parameter(torch.FloatTensor(num_od, hidden_size))
        self.weights_linkflow = Parameter(torch.FloatTensor(T, num_link, hidden_size))
        self.hidden_size = hidden_size
        Initialization([self.weights_demand, self.weights_linkflow])

    def forward(self, demand, X):
        batch_size = len(demand)                                                   # demand: (batch, num_od)
        q = torch.nn.LayerNorm(self.hidden_size)(torch.mm(demand, self.weights_demand))
        q = q.reshape(1, batch_size, 1, -1).repeat(T, 1, 1, 1)                     # (T, batch, 1, hidden)
        w = self.weights_linkflow.reshape(T, 1, num_link, -1).repeat(1, batch_size, 1, 1)
        X = X.permute(1, 2, 0, 3)                                                  # (T, batch, S, num_link)
        x = torch.nn.LayerNorm(self.hidden_size)(torch.matmul(X, w))               # (T, batch, S, hidden)
        scores = torch.matmul(q, x.transpose(2, 3)) / math.sqrt(x.shape[-1])       # e_{s,t}^m (Eq. 30), (T, batch, 1, S)
        att = F.softmax(scores, dim=-1)                                            # lambda_{s,t}^m (Eq. 31)
        x_output = torch.matmul(att, X).reshape(T, batch_size, -1)                 # fused link flows (Eq. 24)
        return att, x_output


class MRLN(Module):
    """Multi-template recurrent computational graph (one demand/route-choice/link chain per template)."""

    def __init__(self, num_link, num_od, hidden_layer, att_hidden_size, hidden_layer2):
        super(MRLN, self).__init__()
        self.num_od = num_od
        # route-choice parameter theta_{s,t}: one scalar per template and step, shared by all OD pairs
        # (block B_theta of Section 6; the name log_theta is kept from the notebook: theta = theta_of(u) in [0.1, 1.5])
        self.log_theta = Parameter(torch.full((S, T), theta_init_raw(THETA_INIT)))   # raw u; theta = theta_of(u)
        self.rec = None                       # when set to {}, forward() records the path times used by the Logit
        # DNN H^(1)_{s,t} of the factor alpha_{s,t} (Eq. 34), part fed with the previous link flows v_{s,t-1} ...
        self.weights1_v = Parameter(torch.FloatTensor(S, num_link, hidden_layer[0]))
        self.weights2_v = Parameter(torch.FloatTensor(S, hidden_layer[0], hidden_layer[1]))
        self.weights3_v = Parameter(torch.FloatTensor(S, hidden_layer[1], num_od))
        self.bias1_v = Parameter(torch.FloatTensor(S, hidden_layer[0]))
        self.bias2_v = Parameter(torch.FloatTensor(S, hidden_layer[1]))
        self.bias3_v = Parameter(torch.FloatTensor(S, num_od))
        Initialization([self.weights1_v, self.weights2_v, self.weights3_v, self.bias1_v, self.bias2_v, self.bias3_v])
        # ... and part fed with the previous OD travel times tau_{s,t-1} (Eq. 33)
        self.weights1_t = Parameter(torch.FloatTensor(S, num_od, hidden_layer[0]))
        self.weights2_t = Parameter(torch.FloatTensor(S, hidden_layer[0], hidden_layer[1]))
        self.weights3_t = Parameter(torch.FloatTensor(S, hidden_layer[1], num_od))
        self.bias1_t = Parameter(torch.FloatTensor(S, hidden_layer[0]))
        self.bias2_t = Parameter(torch.FloatTensor(S, hidden_layer[1]))
        self.bias3_t = Parameter(torch.FloatTensor(S, num_od))
        Initialization([self.weights1_t, self.weights2_t, self.weights3_t, self.bias1_t, self.bias2_t, self.bias3_t])
        self.attention = Attention(num_od, num_link, att_hidden_size)            # block B_lambda of Section 6
        # DNN H^(0)_s that splits the period demand over the T steps (shares p_{s,t}, Eq. 35); with H^(1) = block B_Q
        self.weights_d1 = Parameter(torch.FloatTensor(S, num_od, hidden_layer2[0]))
        self.weights_d2 = Parameter(torch.FloatTensor(S, hidden_layer2[0], hidden_layer2[1]))
        self.weights_d3 = Parameter(torch.FloatTensor(S, hidden_layer2[1], T * num_od))
        self.bias_d1 = Parameter(torch.FloatTensor(S, hidden_layer2[0]))
        self.bias_d2 = Parameter(torch.FloatTensor(S, hidden_layer2[1]))
        self.bias_d3 = Parameter(torch.FloatTensor(S, T * num_od))
        Initialization([self.weights_d1, self.weights_d2, self.weights_d3, self.bias_d1, self.bias_d2, self.bias_d3])

    def forward(self, demand):
        batch_size = demand.shape[0]
        theta = theta_of(self.log_theta)
        link_flow = torch.zeros(S, T, batch_size, num_link)
        demand_pred = torch.zeros(S, T, batch_size, num_od)
        demand_sum = torch.zeros(S, batch_size, num_od)
        for s in range(S):                                                      # one CG per template s
            # share p_{s,t} of each step in the period demand q_bar^m (Eq. 35)
            demand_proportion = torch.nn.LayerNorm(num_od)(demand)
            demand_proportion = F.leaky_relu(torch.mm(demand_proportion, self.weights_d1[s]) + self.bias_d1[s], negative_slope=0.2)
            demand_proportion = F.leaky_relu(torch.mm(demand_proportion, self.weights_d2[s]) + self.bias_d2[s], negative_slope=0.2)
            demand_proportion = F.leaky_relu(torch.mm(demand_proportion, self.weights_d3[s]) + self.bias_d3[s], negative_slope=0.2)
            demand_proportion = demand_proportion.reshape(batch_size, self.num_od, T)
            demand_proportion = torch.softmax(demand_proportion, axis=2)             # (batch, num_od, T)
            demand_initial = demand_proportion * demand.reshape(batch_size, self.num_od, 1)   # p_{s,t} o q_bar^m
            demand_initial = demand_initial.permute(2, 0, 1)                          # (T, batch, num_od)
            # first step (7:00): q_{s,1} = p_{s,1} o q_bar^m, free-flow path times
            demand_st = demand_initial[0]
            theta_st = theta[s, 0]
            LP_s = LP[s]
            link_time_st = free_flow_tt.repeat(batch_size, 1)
            path_time_st = spmm(LPT[s], link_time_st.T).T
            if self.rec is not None:
                self.rec[(s, 0)] = path_time_st.detach().clone()
            link_flow_st, link_time_st = assignment(demand_st, theta_st, path_time_st, LP_s)
            path_time_st = spmm(LPT[s], link_time_st.T).T
            od_travel_time = torch.mean(path_time_st.reshape(batch_size, num_od, k), dim=-1)   # tau_{s,t} (Eq. 33)
            link_flow_s = torch.zeros(T, batch_size, num_link); link_flow_s[0] = link_flow_st
            demand_s = torch.zeros(T, batch_size, num_od); demand_s[0] = demand_st
            demand_sum_s = demand_st
            for t in range(T - 1):
                # recurrent step (Sec. 4.5): factor alpha_{s,t} = 1 + H^(1)(v_{s,t-1}, tau_{s,t-1}) (Eq. 34)
                demand_st_v = torch.nn.LayerNorm(num_link)(link_flow_st)
                demand_st_v = torch.tanh(torch.mm(demand_st_v, self.weights1_v[s]) + self.bias1_v[s])
                demand_st_v = torch.tanh(torch.mm(demand_st_v, self.weights2_v[s]) + self.bias2_v[s])
                demand_st_v = torch.tanh(torch.mm(demand_st_v, self.weights3_v[s]) + self.bias3_v[s])
                demand_st_t = torch.nn.LayerNorm(num_od)(od_travel_time)
                demand_st_t = torch.tanh(torch.mm(demand_st_t, self.weights1_t[s]) + self.bias1_t[s])
                demand_st_t = torch.tanh(torch.mm(demand_st_t, self.weights2_t[s]) + self.bias2_t[s])
                demand_st_t = torch.tanh(torch.mm(demand_st_t, self.weights3_t[s]) + self.bias3_t[s])
                demand_st0 = 0.5 * (demand_st_v + demand_st_t)
                _fac = torch.where(demand_st0 >= 0, 1 + (STEP_HI - 1) * demand_st0, 1 + (1 - STEP_LO) * demand_st0)   # alpha in (0.15, 2)
                # share-based form (Sec. 4.5, 7.3.2): q_{s,t} = alpha_{s,t} o p_{s,t} o q_bar^m, beta_{s,t} = 0
                demand_st = _fac * (demand_initial[t + 1] if SHARE_BASE else demand_st)
                # route choice with theta_{s,t} and the path times pi_{s,t} of the previous step (Eqs. 21-22)
                theta_st = theta[s, t + 1]
                if self.rec is not None:
                    self.rec[(s, t + 1)] = path_time_st.detach().clone()
                link_flow_st, link_time_st = assignment(demand_st, theta_st, path_time_st, LP_s)
                path_time_st = spmm(LPT[s], link_time_st.T).T
                od_travel_time = torch.mean(path_time_st.reshape(batch_size, num_od, k), dim=-1)
                link_flow_s[t + 1] = link_flow_st
                demand_s[t + 1] = demand_st
                demand_sum_s = demand_sum_s + demand_st
            link_flow[s] = link_flow_s
            demand_pred[s] = demand_s
            demand_sum[s] = demand_sum_s
        self.demand_per_template = demand_pred                                         # (S, T, B, num_od)
        link_flow_raw = link_flow                                                      # (S, T, B, num_link)
        att, link_flow = self.attention(demand, link_flow)                             # fused (T, B, num_link), Eq. 24
        demand_pred = torch.matmul(att, demand_pred.permute(1, 2, 0, 3)).reshape(T, batch_size, -1)   # fused OD, Eq. 25
        att = att.reshape(T, batch_size, S)
        return att, demand_sum, demand_pred, link_flow, link_flow_raw


LP = LP0[:S]
LPT = LPT0[:S]
mrln = MRLN(num_link=num_link, num_od=num_od, hidden_layer=[128, 128], att_hidden_size=128, hidden_layer2=[128, 128])
mse_loss = nn.MSELoss()
l1_loss = nn.L1Loss()
theta_params = [mrln.log_theta]
other_params = [p for n, p in mrln.named_parameters() if n != 'log_theta']
optimizer = torch.optim.Adam([{'params': other_params, 'lr': LR_OTHER},
                              {'params': theta_params, 'lr': 0.05}], betas=(0.5, 0.999))
n_epochs = args.epochs
TEST_EVERY = 1

# sensors: 448 links with counts; 45 of them (random, seed 0) are withheld from training for validation
observation_link_number = flow.columns.values.astype(float).astype(int) - 1
num_observed_link = len(observation_link_number)
unobserved_link_idx = np.random.choice(num_observed_link, 45, replace=False)   # 45 withheld = links without sensors
unobserved_link = observation_link_number[unobserved_link_idx]
observed_link_idx = np.delete(np.arange(num_observed_link), unobserved_link_idx)
observed_link = observation_link_number[observed_link_idx]

# ---------------------------------------------------------------------------------------------------------------
# Block-wise training (paper Section 6; code names of the losses -> paper: L1 = L_V, L2 = L_Q, L3 = L_D,
# L5a = L_{Q,T}; weights w1..w5 -> mu_1 = 1, mu_2 = 1, mu_3 = 0.001, mu_4 = 2000)
#   D  : Step 1, demand networks (block B_Q),     L2 + w5 L5a  = mu_2 L_Q + mu_4 L_{Q,T}
#   TH : Step 2, route choice theta (B_theta),    L1 + w3 L3   = mu_1 L_V + mu_3 L_D
#   A  : Step 3, attention (B_lambda; not S = 1), L1 + w3 L3   = mu_1 L_V + mu_3 L_D
#   J  : Step 4, fine-tuning: all blocks, L1 + L2 + w3 L3 + w5 L5a with halved learning rates
#   L5b = per-step loss of the attention-weighted demand; recorded only.
# ---------------------------------------------------------------------------------------------------------------
HISTORY = []
_PROC = psutil.Process()
_pt = (LPT[0] @ free_flow_tt.reshape(-1, 1)).reshape(num_od, k)
_spread = _pt.max(1).values - _pt.min(1).values
_multi = _spread > 1e-6


def fastest_path_share(theta_value):
    p = torch.softmax(-theta_value * _pt[_multi], dim=1)
    return p.max(1).values.median().item()


loss_test_linkflow = torch.zeros(n_epochs)
loss_test_demand = torch.zeros(n_epochs)
loss_test_vdf = torch.zeros(n_epochs)
loss_train = torch.zeros(n_epochs)

ALT_SCHEDULE = [(1, 'D', 20), (1, 'TH', 15), (1, 'A', 15),
                (2, 'D', 8), (2, 'TH', 10), (2, 'A', 8),
                (3, 'D', 5), (3, 'TH', 8), (3, 'A', 6),
                (4, 'J', 5)]                                   # (round, block, maximum epochs); round 4 = together
ALT_MIN, ALT_TOL, ALT_WTOL = 6, 0.01, 0.02                    # block minimum epochs; relative tolerance; weight tolerance  (Section 6)
_GROUPS = {'TH': [n for n, _ in mrln.named_parameters() if n == 'log_theta'],
           'A': [n for n, _ in mrln.named_parameters() if n.startswith('attention.')]}
_GROUPS['D'] = [n for n, _ in mrln.named_parameters() if n not in _GROUPS['TH'] + _GROUPS['A']]
assert all(re.match(r'(weights|bias)(\d_[vt]|_d\d)$', n) for n in _GROUPS['D']), _GROUPS['D']
_PAR = dict(mrln.named_parameters())
_DT = days_train.type.values
_TYPES = ['regular', 'poi', 'stadium', 'both']


def _set_block(block):
    on = sum((_GROUPS[g] for g in ('D', 'TH', 'A')), []) if block == 'J' else _GROUPS[block]
    for n, p in _PAR.items():
        p.requires_grad_(n in on)
    for g in optimizer.param_groups:                           # halve the learning rates in the joint phase
        g['lr'] = g['lr0'] * (0.5 if block == 'J' else 1.0)


def _group_sums():
    with torch.no_grad():
        return {g: float(sum(_PAR[n].abs().sum() for n in names)) for g, names in _GROUPS.items()}


for g in optimizer.param_groups:
    g['lr0'] = g['lr']


def _block_measure(block, rows):
    """Change of the block's own measure (Section 6: the step loss L_{Q,T} for Step 1, theta_{s,t} for Step 2, the mean
    attention weights for Step 3): mean of the last 3 epochs vs the 3 before (None if too few)."""
    if len(rows) < ALT_MIN:
        return None
    h = DataFrame(rows[-6:]); a, b = h.iloc[:3], h.iloc[3:]
    if block == 'D':
        return 1 - b.train_demand_t_a.mean() / max(a.train_demand_t_a.mean(), 1e-12)
    if block == 'TH':
        tc = [c for c in h.columns if c.startswith('theta_')]
        return float(((b[tc].mean() - a[tc].mean()).abs() / a[tc].mean()).median())
    if block == 'A':
        wc = [c for c in h.columns if c.startswith('att_')]
        return float((b[wc].mean() - a[wc].mean()).abs().max())
    return None


def _save_ckpt(i_s, state):
    """Checkpoint at the start of schedule entry i_s (everything needed to replay the rest identically)."""
    os.makedirs(CKPT_DIR, exist_ok=True)
    tmp = CKPT + '.tmp'
    torch.save({'i_s': i_s, 'model': mrln.state_dict(), 'optimizer': optimizer.state_dict(),
                'gen': state['gen'].get_state(), 'epoch': state['epoch'], 'n_upd': state['n_upd'],
                'elapsed': time.time() - state['t0'], 'round_end': state['round_end'], 'STOP_INFO': STOP_INFO,
                'HISTORY': HISTORY, 'losses': [loss_train, loss_test_linkflow, loss_test_demand, loss_test_vdf],
                'S': S, 'n_epochs': n_epochs}, tmp)
    os.replace(tmp, CKPT)


def train(n_epochs):
    w1, w2, w3 = 1, W2_PERIOD, 0.001            # mu_1, mu_2, mu_3 of Eq. 41
    w5 = DEMAND_T_W
    st = {'t0': time.time(), 'gen': torch.Generator().manual_seed(0), 'n_upd': 0, 'epoch': 0, 'round_end': {}}
    STOP_INFO['blocks'] = []; STOP_INFO['rounds'] = []
    sched = [x for x in ALT_SCHEDULE if not (S == 1 and x[1] == 'A')]   # S = 1: one template, no attention block
    STOP_INFO['schedule'] = sched
    i_start = 0
    if os.path.exists(CKPT):                                            # resume at the start of the saved block
        ck = torch.load(CKPT, weights_only=False)
        assert ck['S'] == S and ck['n_epochs'] == n_epochs, 'checkpoint of another run'
        mrln.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer']); st['gen'].set_state(ck['gen'])
        st.update(epoch=ck['epoch'], n_upd=ck['n_upd'], round_end=ck['round_end'], t0=time.time() - ck['elapsed'])
        STOP_INFO.clear(); STOP_INFO.update(ck['STOP_INFO']); HISTORY[:] = ck['HISTORY']
        for dst, src in zip([loss_train, loss_test_linkflow, loss_test_demand, loss_test_vdf], ck['losses']):
            dst.copy_(src)
        i_start = ck['i_s']
        C.log(f'resumed from the checkpoint at schedule entry {i_start} (epoch {st["epoch"]})', PROGRESS)
        del ck
    gen = st['gen']
    for i_s, (rnd, block, n_max) in enumerate(sched):
        if i_s < i_start:
            continue
        _save_ckpt(i_s, st)
        last_in_round = rnd < 4 and (i_s == len(sched) - 1 or sched[i_s + 1][0] != rnd)
        if STOP_INFO.get('skip_to_joint') and block != 'J':
            continue
        _set_block(block)
        g0 = _group_sums(); rows_b = []
        for _ in range(n_max):
            if st['epoch'] >= n_epochs:
                break
            epoch = st['epoch']
            perm = torch.randperm(num_train, generator=gen)
            _sum = np.zeros(7); _nb = 0
            _wsum = np.zeros((len(_TYPES), S)); _wn = np.zeros(len(_TYPES)); _wt = np.zeros((T, S))
            _tpl_err = np.zeros(S); _tpl_n = 0.0; _fus_err = 0.0
            for b0 in range(0, num_train, BATCH):
                idx = perm[b0:b0 + BATCH]
                mrln.zero_grad(set_to_none=True)
                demand_input = train_input[idx]
                link_flow_output = train_output[:, idx, :]
                att_pred, demand_sum, demand_pred, link_flow_pred, link_flow_raw = mrln(demand_input)
                flow_with_sensors = link_flow_output[:, :, observed_link_idx]
                link_flow_loss = mse_loss(link_flow_pred[:, :, observed_link], flow_with_sensors)               # L1 = L_V (Eq. 36)
                demand_loss = mse_loss(demand_sum, demand_input.reshape(1, demand_input.shape[0], -1).repeat(S, 1, 1))  # L2 = L_Q (Eq. 37)
                # integral of the BPR time from 0 to v (Beckmann term): t0 v + 0.15 t0 (4v)^4 v / (5 C^4)
                vdf_estimate = link_flow_pred * free_flow_tt + 0.03 * (FLOW_TO_HOURLY ** 4) * (link_flow_pred ** 5) * free_flow_tt / (link_capacity ** 4)
                vdf_loss = torch.mean(torch.sum(vdf_estimate, dim=-1))                                          # L3 = L_D (Eq. 38)
                ref_t = demand_train_reference[:, idx, :]
                dpt = mrln.demand_per_template                                                                  # (S, T, B, num_od)
                dem_t_a = mse_loss(dpt, ref_t[None].expand_as(dpt))                  # L5a = L_{Q,T} (Eq. 40): each template vs the sample
                dem_t_b = mse_loss(demand_pred, ref_t)                               # L5b: fused demand (recorded only)
                if block == 'D':
                    total_loss = w2 * demand_loss + w5 * dem_t_a
                elif block in ('TH', 'A'):
                    total_loss = w1 * link_flow_loss + w3 * vdf_loss
                else:
                    total_loss = w1 * link_flow_loss + w2 * demand_loss + w3 * vdf_loss + w5 * dem_t_a
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=[p for p in mrln.parameters() if p.grad is not None], max_norm=1, norm_type=2)
                optimizer.step()
                _sum += [total_loss.item(), link_flow_loss.item(), demand_loss.item(), vdf_loss.item(), 0.0, dem_t_a.item(), dem_t_b.item()]; _nb += 1
                with torch.no_grad():
                    a_ = att_pred.mean(0).numpy()                                  # (B, S), mean over time steps
                    for j_, d_ in enumerate(idx.tolist()):
                        i_ = _TYPES.index(_DT[d_]); _wsum[i_] += a_[j_]; _wn[i_] += 1
                    _wt += att_pred.sum(1).numpy()                                 # (T, S)
                    o_ = train_output[:, idx, :]; m_ = o_ > 1
                    for s_ in range(S):
                        p_ = link_flow_raw[s_][:, :, observation_link_number]
                        _tpl_err[s_] += float(((p_ - o_).abs() / o_.clamp(min=1))[m_].sum())
                    _fus_err += float(((link_flow_pred[:, :, observation_link_number] - o_).abs() / o_.clamp(min=1))[m_].sum())
                    _tpl_n += float(m_.sum())
            st['n_upd'] += _nb
            tr_total, tr_link, tr_demand, tr_vdf, _, tr_dem_t_a, tr_dem_t_b = _sum / _nb
            with torch.no_grad():                                                  # test pass every epoch
                att_test_pred, demand_test_sum_pred, demand_test_pred, link_flow_test_pred, _ = mrln(test_input)
                loss_test_linkflow[epoch] = mse_loss(test_output, link_flow_test_pred[:, :, observation_link_number]).item()
                loss_test_demand[epoch] = mse_loss(demand_reference, demand_test_pred).item()
                vdf_test_estimate = link_flow_test_pred * free_flow_tt + 0.03 * (FLOW_TO_HOURLY ** 4) * (link_flow_test_pred ** 5) * free_flow_tt / (link_capacity ** 4)
                loss_test_vdf[epoch] = torch.mean(torch.sum(vdf_test_estimate, dim=-1)).item(); _mape_now(link_flow_test_pred)
            loss_train[epoch] = tr_total
            with torch.no_grad():
                th = theta_of(mrln.log_theta)
                share = fastest_path_share(th.median())
                wt = _wt / num_train                                                 # (T, S) mean weights
                fused_vot = float(60 * (wt * th.numpy().T).sum(1).mean())
                _row = {'epoch': epoch, 'round': rnd, 'block': block, 'elapsed_s': time.time() - st['t0'],
                        'n_updates': st['n_upd'], 'train_total': tr_total, 'train_link': tr_link,
                        'train_demand': tr_demand, 'train_vdf': tr_vdf, 'train_anchor': 0.0, 'w_anchor': None,
                        'train_demand_t': tr_dem_t_b, 'train_demand_t_a': tr_dem_t_a,
                        'test_link': float(loss_test_linkflow[epoch]), 'test_demand': float(loss_test_demand[epoch]),
                        'test_vdf': float(loss_test_vdf[epoch]), 'fastest_path_share': share,
                        'fused_vot': fused_vot, 'train_sensor_mape_fused': _fus_err / _tpl_n,
                        'rss_gb': _PROC.memory_info().rss / 1e9}
                _row.update({f'train_sensor_mape_T{s_+1}': _tpl_err[s_] / _tpl_n for s_ in range(S)})
                _row.update({f'att_{t_}_T{s_+1}': float(_wsum[i_, s_] / max(_wn[i_], 1)) for i_, t_ in enumerate(_TYPES) for s_ in range(S)})
                _row.update({f'theta_s{s_+1}_t{t_+1}': float(th[s_, t_]) for s_ in range(th.shape[0]) for t_ in range(th.shape[1])})
                HISTORY.append(_row); _progress(_row); rows_b.append(_row)
                msg_ = (f"{time.strftime('%H:%M:%S')}   [round {rnd} block {block} epoch {len(rows_b)}/{n_max}] L5a {tr_dem_t_a:.4f} L5b {tr_dem_t_b:.4f} "
                        f"fused VOT {fused_vot:.1f} | train sensor MAPE fused {_fus_err / _tpl_n:.1%}, alone "
                        + ' / '.join(f'T{s_+1} {_tpl_err[s_] / _tpl_n:.1%}' for s_ in range(S))
                        + ' | weights stadium days ' + ' / '.join(f"{_row[f'att_stadium_T{s_+1}']:.2f}" for s_ in range(S)))
                try:
                    with open(PROGRESS, 'a') as f_:
                        f_.write(msg_ + chr(10))
                except OSError:
                    pass
                print(msg_, flush=True)
            st['epoch'] += 1
            STOP_INFO['epochs_run'] = st['epoch']
            ch_ = _block_measure(block, rows_b)
            if ch_ is not None and ch_ < (ALT_WTOL if block == 'A' else ALT_TOL):
                break
        g1 = _group_sums()
        STOP_INFO['blocks'].append({'round': rnd, 'block': block, 'epochs': len(rows_b), 'max': n_max,
                                    'last_change': _block_measure(block, rows_b),
                                    'group_abs_sum_before': g0, 'group_abs_sum_after': g1})
        if last_in_round and rows_b:                                               # end of a round: compare with the last one
            r_ = rows_b[-1]; st['round_end'][rnd] = r_
            info_ = {'round': rnd, 'epoch_end': st['epoch'], 'fused_vot': r_['fused_vot'], 'train_link': r_['train_link'],
                     'train_sensor_mape_alone': [r_[f'train_sensor_mape_T{s_+1}'] for s_ in range(S)]}
            if rnd - 1 in st['round_end']:
                p_ = st['round_end'][rnd - 1]
                tc = [c for c in r_ if c.startswith('theta_')]; wc = [c for c in r_ if c.startswith('att_')]
                d_th = float(np.median([abs(r_[c] - p_[c]) / p_[c] for c in tc]))
                d_l = 1 - r_['train_link'] / p_['train_link']
                d_w = max(abs(r_[c] - p_[c]) for c in wc)
                info_.update({'theta_change': d_th, 'link_drop': d_l, 'weight_change': d_w})
                if d_th < ALT_TOL and d_l < ALT_TOL and d_w < ALT_WTOL:
                    STOP_INFO['skip_to_joint'] = True
            STOP_INFO['rounds'].append(info_)
            print('[round end]', info_, flush=True)
    for p in mrln.parameters():
        p.requires_grad_(True)
    for g in optimizer.param_groups:
        g['lr'] = g['lr0']
    STOP_INFO['reason'] = 'schedule finished' + (' (rounds converged early)' if STOP_INFO.get('skip_to_joint') else '')


C.log(f'MTCG S={S}: training starts (threads {torch.get_num_threads()})', PROGRESS)
train(n_epochs)
n_run = STOP_INFO['epochs_run']
loss_train, loss_test_linkflow, loss_test_demand, loss_test_vdf = (loss_train[:n_run], loss_test_linkflow[:n_run], loss_test_demand[:n_run], loss_test_vdf[:n_run])
print('stop:', STOP_INFO)

# ---------------------------------------------------------------------------------------------------------------
# Inference, recorded path states, inference timing
# ---------------------------------------------------------------------------------------------------------------
with torch.no_grad():
    att_pred, demand_test_sum_pred, demand_test_pred, link_flow_test_pred, link_flow_test_raw = mrln(test_input)
    att_train, demand_train_sum_pred, demand_train_pred, link_flow_train_pred, link_flow_train_raw = mrln(train_input)
    mrln.rec = {}
    mrln(test_input)                       # same result; records the path times fed to the Logit
    rec_test = mrln.rec; mrln.rec = None
_inf = []
with torch.no_grad():
    for _ in range(10):
        _t = time.time(); mrln(test_input); _inf.append(time.time() - _t)
print('inference time per full sequence: %.3f +- %.3f s' % (np.mean(_inf), np.std(_inf)))

# error table: sensors used in training, withheld sensors, all sensors, OD demand (20 test days)
Error = torch.zeros(4, 3)
Error[0, 0] = mse_loss(test_output[:, :, observed_link_idx], link_flow_test_pred[:, :, observed_link]) ** 0.5
Error[0, 1] = l1_loss(test_output[:, :, observed_link_idx], link_flow_test_pred[:, :, observed_link])
E1 = torch.abs(test_output[:, :, observed_link_idx] - link_flow_test_pred[:, :, observed_link]) / test_output[:, :, observed_link_idx]
Error[0, 2] = torch.mean(E1[E1 < 100])
Error[1, 0] = mse_loss(test_output[:, :, unobserved_link_idx], link_flow_test_pred[:, :, unobserved_link]) ** 0.5
Error[1, 1] = l1_loss(test_output[:, :, unobserved_link_idx], link_flow_test_pred[:, :, unobserved_link])
E2 = torch.abs(test_output[:, :, unobserved_link_idx] - link_flow_test_pred[:, :, unobserved_link]) / test_output[:, :, unobserved_link_idx]
Error[1, 2] = torch.mean(E2[E2 < 100])
Error[2, 0] = mse_loss(test_output, link_flow_test_pred[:, :, observation_link_number]) ** 0.5
Error[2, 1] = l1_loss(test_output, link_flow_test_pred[:, :, observation_link_number])
E3 = torch.abs(test_output - link_flow_test_pred[:, :, observation_link_number]) / test_output
Error[2, 2] = torch.mean(E3[E3 < 100])
Error[3, 0] = mse_loss(demand_reference, demand_test_pred) ** 0.5
Error[3, 1] = l1_loss(demand_reference, demand_test_pred)
E4 = torch.abs(demand_reference - demand_test_pred) / demand_reference
Error[3, 2] = torch.mean(E4[E4 < 100])
Error = Error.detach().numpy()
Error[:, :2] = np.floor(100 * Error[:, :2]) / 100
Error[:, 2] = np.floor(10000 * Error[:, 2]) / 10000
Error = DataFrame(Error, index=['With sensors', 'Without sensors', 'Total error', 'OD demand'], columns=['RMSE', 'MAE', 'MAPE'])

# ---------------------------------------------------------------------------------------------------------------
# Exports (same files as the experiment)
# ---------------------------------------------------------------------------------------------------------------
theta_st = theta_of(mrln.log_theta).detach().numpy()                  # (S, T), 1/minute
DataFrame(theta_st, index=[f'template {s+1}' for s in range(S)],
          columns=[f't{t+1}' for t in range(T)]).to_csv(os.path.join(OUT, 'theta_st.csv'))
hist = DataFrame(HISTORY); hist.to_csv(os.path.join(OUT, 'training_history.csv'), index=False)
DataFrame(att_pred[:, 0, :].numpy()).to_csv(os.path.join(OUT, 'att.csv'), index=False)
_a = torch.sum(demand_test_pred, axis=-1).numpy(); _b = torch.sum(demand_reference, axis=-1).numpy()
_c = link_flow_test_pred[:, :, observation_link_number].sum(axis=-1).numpy(); _d = test_output.sum(axis=-1).numpy()
DataFrame(np.hstack([_a, _b, _c, _d])).to_csv(os.path.join(OUT, 'estimation2.csv'), index=False)
DataFrame(loss_train.numpy()).to_excel(os.path.join(OUT, 'loss_train.xlsx'), index=False)
DataFrame((loss_test_linkflow + 0.5 * loss_test_demand + 0.01 * loss_test_vdf).numpy()).to_excel(
    os.path.join(OUT, 'loss_test.xlsx'), index=False)
Error.to_csv(os.path.join(OUT, 'error_table.csv'))
np.savez_compressed(os.path.join(OUT, 'predictions.npz'),
    observation_link_number=observation_link_number, observed_link_idx=observed_link_idx,
    unobserved_link_idx=unobserved_link_idx,
    test_att=att_pred.numpy(), test_link_obs=test_output.numpy(),
    test_link_pred=link_flow_test_pred.numpy(), test_link_pred_per_template=link_flow_test_raw.numpy(),
    test_demand_pred=demand_test_pred.numpy(), test_demand_ref=demand_reference.numpy(),
    test_demand_sum_per_template=demand_test_sum_pred.numpy(), test_input=test_input.numpy(),
    train_att=att_train.numpy(), train_link_obs=train_output.numpy(),
    train_link_pred=link_flow_train_pred.numpy(), train_link_pred_per_template=link_flow_train_raw.numpy(),
    train_demand_pred=demand_train_pred.numpy(), train_demand_ref=demand_train_reference.numpy(),
    train_input=train_input.numpy())
# path times and route shares at every (template, step), test days
_pt = np.stack([np.stack([rec_test[(s, t)].numpy() for t in range(T)]) for s in range(S)])   # (S, T, B, num_path)
_u = -theta_st[:, :, None, None] * _pt
_u = _u.reshape(S, T, _pt.shape[2], num_od, k); _u -= _u.max(-1, keepdims=True)
_sh = np.exp(_u); _sh /= _sh.sum(-1, keepdims=True)
np.savez_compressed(os.path.join(OUT, 'path_state.npz'), path_time=_pt.astype(np.float32),
                    path_share=_sh.reshape(_pt.shape).astype(np.float32), theta_st=theta_st, k=k)
for s, pl in enumerate([path_last1, path_last2, path_last3][:S]):     # column order of LP / path_state
    pl[['o_zone_id', 'd_zone_id', 'path_id']].to_csv(os.path.join(OUT, f'path_order_T{s+1}.csv'), index=False)

_ep = np.diff(np.concatenate([[0.0], hist['elapsed_s'].values]))
run_info = {
    'data': DAYS, 'batch_days': BATCH, 'n_train_days': int(num_train), 'n_test_days': int(test_input.shape[0]),
    'n_updates': int(hist['n_updates'].iloc[-1]), 'window': '7:00-9:45',
    'anchor': ANCHOR, 'anchor_share': ANCHOR_SHARE, 'w_anchor': W_ANCHOR, 'theta_bounds': [THETA_LO, THETA_HI],
    'step_range': [STEP_LO, STEP_HI], 'theta_init_setting': THETA_INIT, 'flow_to_hourly': FLOW_TO_HOURLY,
    'true_demand': TRUE_DEMAND, 'demand_t_share': DEMAND_T_SHARE, 'demand_t_w': DEMAND_T_W, 'w2_period': W2_PERIOD,
    'share_base': SHARE_BASE, 'stop_info': STOP_INFO, 'S': S, 'T': T, 'k': k, 'n_epochs': n_epochs, 'seed': 0,
    'lr_other': LR_OTHER, 'lr_theta': 0.05, 'theta_init': 1.0, 'run_tag': RUN_TAG + '_alt',
    'epoch_time_mean_s': float(_ep.mean()), 'epoch_time_sd_s': float(_ep.std()),
    'total_training_time_s': float(hist['elapsed_s'].iloc[-1]),
    'inference_time_full_sequence_mean_s': float(np.mean(_inf)), 'inference_time_full_sequence_sd_s': float(np.std(_inf)),
    'inference_time_per_step_mean_s': float(np.mean(_inf) / T),
    'peak_rss_gb_during_training': float(hist['rss_gb'].max()),
    'machine_ram_gb': psutil.virtual_memory().total / 1e9, 'cpu_count': psutil.cpu_count(),
    'torch_threads': torch.get_num_threads(),
    'python': sys.version.split()[0], 'torch': torch.__version__, 'platform': platform.platform(),
    'theta_per_template_mean': theta_st.mean(1).round(4).tolist(),
    'theta_min': float(theta_st.min()), 'theta_median': float(np.median(theta_st)), 'theta_max': float(theta_st.max()),
    'theta_fused_by_time_first_test_day': (att_pred[:, 0, :].numpy() * theta_st.T).sum(1).round(4).tolist(),
    'fastest_path_share_final': float(hist['fastest_path_share'].iloc[-1]),
    'error_table': Error.reset_index().to_dict(orient='list'),
}
json.dump(run_info, open(os.path.join(OUT, 'run_info.json'), 'w'), indent=1)
print(DataFrame(theta_st, index=[f'template {s+1}' for s in range(S)], columns=[f't{t+1}' for t in range(T)]).round(4))
print(Error)

# per-day outputs: error by test day, attention weights vs the true event sizes a2, a3, demand fit
days_train.to_csv(os.path.join(OUT, 'days_train.csv'), index=False)
days_test.to_csv(os.path.join(OUT, 'days_test.csv'), index=False)


def _err(o, p):
    e = (o - p).abs() / o
    return float(((o - p) ** 2).mean() ** 0.5), float((o - p).abs().mean()), float(e[e < 100].mean())


_P = link_flow_test_pred[:, :, observation_link_number]
_rows = []
for j, r in days_test.reset_index(drop=True).iterrows():
    for name, o, p in (('With sensors', test_output[:, j, observed_link_idx], _P[:, j, observed_link_idx]),
                       ('Without sensors', test_output[:, j, unobserved_link_idx], _P[:, j, unobserved_link_idx]),
                       ('Total error', test_output[:, j, :], _P[:, j, :]),
                       ('OD demand', demand_reference[:, j, :], demand_test_pred[:, j, :])):
        rm, ma, mp = _err(o, p)
        _rows.append({'day': r.day, 'type': r.type, 'a2': r.a2, 'a3': r.a3, 'item': name, 'RMSE': rm, 'MAE': ma, 'MAPE': mp})
DataFrame(_rows).to_csv(os.path.join(OUT, 'error_by_day.csv'), index=False)
_clock = C.CLOCK


def _att_table(att, info):
    a_ = att.numpy()                                                          # (T, B, S)
    out = []
    for j, r in info.reset_index(drop=True).iterrows():
        for t in range(T):
            row = {'day': r.day, 'type': r.type, 'a2': r.a2, 'a3': r.a3, 't': t + 1, 'clock': _clock[t]}
            row.update({f'w_T{s+1}': float(a_[t, j, s]) for s in range(S)})
            row['theta_fused'] = float((a_[t, j, :] * theta_st[:, t]).sum())
            out.append(row)
    return DataFrame(out)


_att_table(att_pred, days_test).to_csv(os.path.join(OUT, 'attention_by_day.csv'), index=False)
_att_table(att_train, days_train).to_csv(os.path.join(OUT, 'attention_by_day_train.csv'), index=False)
_dm = demand_test_pred.sum(-1).numpy(); _dr = demand_reference.sum(-1).numpy()          # (T, B)
_fit = []
for j, r in days_test.reset_index(drop=True).iterrows():
    for t in range(T):
        _fit.append({'day': r.day, 'type': r.type, 'a2': r.a2, 'a3': r.a3, 'clock': _clock[t], 'model': float(_dm[t, j]), 'true': float(_dr[t, j])})
DataFrame(_fit).to_csv(os.path.join(OUT, 'demand_fit_by_day.csv'), index=False)
with torch.no_grad():
    mrln(test_input)
    _dpt = mrln.demand_per_template.sum(-1).numpy()                          # (S, T, B)
np.savez_compressed(os.path.join(OUT, 'template_demand_test.npz'), per_template=_dpt,
                    reference=Q_TEMPLATE[:S].sum(-1).numpy())
if os.path.exists(CKPT):
    os.remove(CKPT)
C.log(f'MTCG S={S} finished: {n_run} epochs, files in {os.path.relpath(OUT, C.ROOT)}', PROGRESS)
