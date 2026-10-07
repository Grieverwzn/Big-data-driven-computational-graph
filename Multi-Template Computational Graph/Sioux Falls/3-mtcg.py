"""
Sioux Falls training of the Multi-Template Computational Graph (MTCG), paper "Can Contextual Archetypes Explain Daily
Traffic Variation? A Multi-Template Computational Graph with Attention-Based Fusion" (Section 7.2; model in Sections 4.4-4.7:
attention fusion 4.4, recurrent dynamic extension 4.5, losses 4.6).

Authors:
    Xin (Bruce) Wu, Department of Civil and Environmental Engineering, Villanova University, PA, USA
    Feng Shao, School of Mathematics, China University of Mining and Technology, China

Contact: xwu03@villanova.edu (Villanova University), xinwu8592@gmail.com (personal)

MIT License
Copyright (c) 2026 Xin (Bruce) Wu, Feng Shao

Usage  : python "3-mtcg.py" --S 5 --seed 42                  (standard setting, paper Table 6, rows MTCG S = 1..5)
         python "3-mtcg.py" --S 5 --seed 43 --od-weighted    (OD-weighted setting, paper Section 7.2.3)
         python "3-mtcg.py" --S 1 --seed 42 --layout n23_L0 --no-test-loss
                                                             (sensor-coverage run, paper Section 7.2.5, Table 7)
         python "3-mtcg.py" --S 1 --seed 42 --iters 5        (quick check that the script runs; a few seconds)
         options: --S 1..5 (number of templates), --seed (42, 43, 44 in the paper), --iters (default 1000),
                  --threads (torch CPU threads, default 8), --layout (a sensor layout of data/sensor_layouts.csv;
                  default: the 66 links with sensors of paper Fig. 13), --no-test-loss (do not evaluate the test
                  loss after every iteration; faster, used for the coverage runs; loss_test.xlsx is not written)
         Needs data/ (1-data_generation.py), templates/lp_S<S>.npz (2-template_generation.py) and
         sf_common.py next to this script (numpy, pandas, torch, openpyxl). Runs on the CPU; about 1 min (S = 1) to
         4-9 min (S = 5) with 8 threads, peak memory below 2 GB.

Model (one forward pass, for every template s = 1..S; all quantities per sample m; settings of paper Section 7.2.2).
Paper symbol -> code name:
  qbar^m (aggregate OD demand, 96 values)       demand_total / demand (model input)
  delta_s (path-link incidence of template s)    LP[s] (76 links x 480 paths, i.e. delta_s transposed)
  theta_{s,t} (Logit parameter)                  |self.theta[s, t]| (one value per OD pair; Xavier initialization)
  H^(0)_s  (demand of the first time step)       weights_d1..d3 / bias_d1..d3
  H^(1)_{s,t} (scaling factor h_{s,t}, alpha_{s,t} = 1 + 0.5 h_{s,t})
                                                 weights1_v..3_v (link flows), weights1_t..3_t (OD times), weights_fusion
  H^(2)_{s,t} (shift term beta_{s,t})            weights1_v2 / bias1_v2
  attention weights lambda^m_{s,t}              Attention module (d_k = 128), returned as att
  mu_1, mu_2, mu_3, mu_4 (loss weights)          w1, w2, w3, w4
  L_V, L_Q, L_D, L_{Q,T} (losses)                L_V, L_Q, L_D, L_q in losses()
Steps:
  1. Initial demand. H^(0)_s (96 -> 512 -> 512 -> 8x96, LeakyReLU 0.2, softmax over the 8 steps of each OD pair)
     splits qbar^m over the time steps; the share of the first step times qbar^m is q_{s,1}.
  2. Recursive loading. At each step t, the demand q_{s,t} is loaded on the template's route set by the Logit model
     (parameter |theta_{s,t,w}|, one per template, step and OD pair) with path costs from the BPR link times of the
     previous step (free-flow times at t = 1), giving link flows v_{s,t} and link times.
  3. Demand recursion (Section 4.5). q_{s,t+1} = alpha_{s,t} * q_{s,t} + beta_{s,t}, alpha_{s,t} = 1 + 0.5 h_{s,t} in
     (0.5, 1.5): h_{s,t} in (-1, 1) comes from H^(1)_{s,t} (two tanh networks 512 x 512 x 96 on the link flows and on
     the mean OD path times, joined by one linear layer and tanh); beta_{s,t} comes from H^(2)_{s,t} (one linear layer on
     q_{s,t}, LeakyReLU 0.2).
  4. Attention fusion (Section 4.4). The query is the embedded aggregate OD demand, the keys are the embedded link flows
     of each template at each step (d_k = 128, layer normalization); softmax weights over the S templates fuse the link
     flows and the per-step OD demand.
Losses (Section 4.6): L = mu_1 L_V + mu_2 L_Q + mu_3 L_D + mu_4 L_{Q,T} with
  L_V      mean squared error of the fused link flows on the 66 links with sensors (training samples),
  L_Q      period-level OD demand: the sum over the 8 steps of q_{s,t} equals qbar^m, for every template,
  L_D      VDF term, mean over samples of sum_a [v_a t0_a + 0.03 v_a^5 t0_a / c_a^4] (BPR integral),
  L_{Q,T}  time-step OD demand error (not used, mu_4 = 0; the time-step demand is never given to the model).
Training: Adam (lr 0.001, betas (0.9, 0.999)), batches of 64 of the 400 training samples, 1000 iterations, gradient
clipping at norm 1. Standard setting: mu_1 = 1, mu_2 = 0.8, mu_3 = 0.01, mu_4 = 0, constant learning rate.
OD-weighted setting (--od-weighted, Section 7.2.3): mu_2 = 5 and cosine learning-rate decay from 0.001 to 10% of it.
The test loss is evaluated on the 100 test samples after every iteration (only recorded, for the convergence figure).
The reported test OD demand is max(0, q) (the additive term of the recursion can make a few estimates negative).

Output: results/S<S>_seed<seed>[_odw]/  (coverage runs: results/coverage/<layout>_s<seed>_S<S>/)
  link_flow_estimation.xlsx   test link flows: Estimation, Observation (8 steps x 100 samples x 76 links, step-major)
  od_demand_estimation.xlsx   test OD demand: Estimation, Reference (8 steps x 100 samples x 96 OD pairs)
  loss_train.xlsx, loss_test.xlsx   losses at each iteration: link_flow (L_V), demand (L_Q + L_{Q,T}), vdf (L_D), total
  theta_attention.npz         |theta| (S x 8 x 96), test attention weights (8 x 100 x S), links with / without sensors
  error_table.csv             test RMSE / MAE / MAPE of links with sensors, without sensors, all links and OD demand
  run_info.json               settings, error table, training-sample sensor MAPE, theta and attention summaries, run time
"""
import os, json, math, time, random, argparse
import numpy as np, pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from sf_common import (DATA, TEMPLATES, K, T, NTRAIN, num_link, num_od, link_attributes, OBSERVED, UNOBSERVED,
                       load_data, run_dir)

ap = argparse.ArgumentParser(description='Train the MTCG on the Sioux Falls data')
ap.add_argument('--S', type=int, required=True, choices=range(1, 6), help='number of templates (1..5)')
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--od-weighted', action='store_true', help='mu_2 = 5 and cosine learning-rate decay to 10%% (Section 7.2.3)')
ap.add_argument('--iters', type=int, default=1000, help='training iterations (batches)')
ap.add_argument('--threads', type=int, default=8, help='torch CPU threads')
ap.add_argument('--layout', default=None, help='sensor layout name in data/sensor_layouts.csv (Section 7.2.5), e.g. n23_L0')
ap.add_argument('--no-test-loss', action='store_true', help='skip the test loss after every iteration (coverage runs)')
args = ap.parse_args()
torch.set_num_threads(args.threads)
T_START = time.time()

# ---- settings ----
S, SEED, k = args.S, args.seed, K
w1, w2, w3, w4 = 1.0, (5.0 if args.od_weighted else 0.8), 0.01, 0.0   # loss weights mu_1..mu_4 of L_V, L_Q, L_D, L_{Q,T}
LR, BETAS, LR_MIN_FRAC = 0.001, (0.9, 0.999), 0.1
batch_size, n_iters, sample_size = 64, args.iters, NTRAIN
output_dir = run_dir(S, SEED, args.od_weighted, args.layout)
os.makedirs(output_dir, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---- data: aggregate OD demand (input), per-step OD demand (evaluation only), link flows ----
od_demand, flow = load_data()                                                        # (12, 500, 96), (12, 500, 76)
demand_total = torch.from_numpy(od_demand[:T].sum(axis=0)).to(torch.float32)       # (500, 96) aggregate demand qbar^m
demand_reference = torch.from_numpy(od_demand[:T]).to(torch.float32)                # (T, 500, 96)
link_flow = torch.from_numpy(flow[:T]).to(torch.float32)                             # (T, 500, 76)
free_flow_tt = torch.from_numpy(link_attributes['free flow travel time'].values).to(torch.float32)
link_capacity = torch.from_numpy(link_attributes['link capacity'].values).to(torch.float32)
observed_link, unobserved_link = OBSERVED, UNOBSERVED                               # 1-based link numbers
if args.layout:                                     # sensor-coverage run (Section 7.2.5): another set of links with sensors
    lay = pd.read_csv(os.path.join(DATA, 'sensor_layouts.csv')).set_index('name').loc[args.layout]
    observed_link = np.array(sorted(int(a) for a in lay['observed'].split(',')))
    unobserved_link = np.setdiff1d(np.arange(1, num_link + 1), observed_link)

train_input, train_output = demand_total[:NTRAIN], link_flow[:, :NTRAIN]           # first 400 samples
test_input, test_output = demand_total[NTRAIN:], link_flow[:, NTRAIN:]             # last 100 samples

# ---- templates (route sets): LP[s] is the link-path incidence matrix (76 x 480) of template s (delta_s transposed) ----
LP = np.load(os.path.join(TEMPLATES, f'lp_S{S}.npz'))['LP'][:S]                    # S = 1: 5 free-flow shortest paths
LP = torch.from_numpy(LP).to(torch.float32)
# The original notebook drew (S-1) x 96 random vectors of 76 values while building placeholder route sets (replaced by
# the templates above). The same numbers are drawn here so that the training batches match the paper's runs exactly.
np.random.normal(1, 1, (S - 1) * num_od * num_link)


def BPR(link_flow, free_flow_tt, link_capacity):
    """BPR link travel time"""
    return free_flow_tt * (1 + 0.15 * (link_flow / link_capacity) ** 4)


def assignment(demand, theta, path_time, LP):
    """Logit loading of one step: OD demand (batch x 96) -> path choice (Logit, parameter theta per OD pair, path
    times path_time) -> path flows -> link flows (batch x 76) and BPR link times"""
    batch_size = demand.shape[0]
    if path_time.dim() == 1:
        path_time = path_time.repeat(batch_size, 1)
    demand_expand = demand.reshape(-1, 1).repeat(1, k).reshape(batch_size, -1)      # (batch, num_path)
    theta_expand = theta.repeat(k, 1).transpose(0, 1).reshape(1, -1)                # (1, num_path)
    tp0 = (-path_time * theta_expand).reshape(batch_size, -1, k)                    # (batch, num_od, k)
    tp = math.e ** tp0                                                              # e^(-theta * path time)
    P0 = torch.max(tp, (10 ** -10) * torch.ones(tp.shape))                          # lower bound 1e-10
    P0 = P0 / torch.sum(P0, dim=2).view(batch_size, -1, 1)                          # path choice proportions
    path_flow = demand_expand * P0.view(batch_size, -1)                             # (batch, num_path)
    link_flow = torch.mm(path_flow, LP.T)                                           # (batch, num_link)
    return link_flow, BPR(link_flow, free_flow_tt, link_capacity)


def Initialization(parameters):
    for para in parameters:
        nn.init.kaiming_normal_(para)


class Attention(nn.Module):
    """Attention fusion of the S templates (Section 4.4): query = embedded aggregate OD demand, keys = embedded link
    flows of each template at each step; hidden_size = d_k"""

    def __init__(self, num_od, num_link, hidden_size):
        super(Attention, self).__init__()
        self.weights_demand = Parameter(torch.FloatTensor(num_od, hidden_size))
        self.weights_linkflow = Parameter(torch.FloatTensor(T, num_link, hidden_size))
        self.hidden_size = hidden_size
        Initialization([self.weights_demand, self.weights_linkflow])

    def forward(self, demand, X):
        batch_size = len(demand)                                                    # demand: (batch, num_od)
        q = F.layer_norm(torch.mm(demand, self.weights_demand), (self.hidden_size,))   # query (batch, d_k)
        q = q.reshape(1, batch_size, 1, -1).repeat(T, 1, 1, 1)                      # (T, batch, 1, d_k)
        w = self.weights_linkflow.reshape(T, 1, num_link, -1).repeat(1, batch_size, 1, 1)   # (T, batch, num_link, d_k)
        X = X.permute(1, 2, 0, 3)                                                   # (S, T, batch, L) -> (T, batch, S, L)
        x = F.layer_norm(torch.matmul(X, w), (self.hidden_size,))                   # keys (T, batch, S, d_k)
        scores = torch.matmul(q, x.transpose(2, 3)) / math.sqrt(x.shape[-1])        # (T, batch, 1, S)
        att = F.softmax(scores, dim=-1)                                             # attention weights over templates
        x_output = torch.matmul(att, X).reshape(T, batch_size, -1)                  # fused link flows (T, batch, L)
        return att, x_output


class MTCG(nn.Module):
    """Multi-Template Computational Graph (paper Sections 4.4-4.7). hidden_layer: hidden sizes of the two networks of
    H^(1)_{s,t}; att_hidden_size: d_k; hidden_layer2: hidden sizes of H^(0)_s (split of qbar^m over the time steps)"""

    def __init__(self, num_link, num_od, hidden_layer, att_hidden_size, hidden_layer2):
        super(MTCG, self).__init__()
        self.num_od = num_od
        # theta_{s,t}: Logit parameter, one per template, step and OD pair (|theta| is used)
        self.theta = Parameter(torch.FloatTensor(S, T, num_od))
        nn.init.xavier_uniform_(self.theta)
        # H^(0)_s: demand of the first time step = split of the aggregate demand over the T steps (three-layer DNN)
        self.weights_d1 = Parameter(torch.FloatTensor(S, num_od, hidden_layer2[0]))
        self.weights_d2 = Parameter(torch.FloatTensor(S, hidden_layer2[0], hidden_layer2[1]))
        self.weights_d3 = Parameter(torch.FloatTensor(S, hidden_layer2[1], T * num_od))
        self.bias_d1 = Parameter(torch.FloatTensor(S, hidden_layer2[0]))
        self.bias_d2 = Parameter(torch.FloatTensor(S, hidden_layer2[1]))
        self.bias_d3 = Parameter(torch.FloatTensor(S, T * num_od))
        Initialization([self.weights_d1, self.weights_d2, self.weights_d3, self.bias_d1, self.bias_d2, self.bias_d3])
        # H^(1)_{s,t}, network 1: on the link flows of the previous step (weights shared by all steps)
        self.weights1_v = Parameter(torch.FloatTensor(S, num_link, hidden_layer[0]))
        self.weights2_v = Parameter(torch.FloatTensor(S, hidden_layer[0], hidden_layer[1]))
        self.weights3_v = Parameter(torch.FloatTensor(S, hidden_layer[1], num_od))
        self.bias1_v = Parameter(torch.FloatTensor(S, hidden_layer[0]))
        self.bias2_v = Parameter(torch.FloatTensor(S, hidden_layer[1]))
        self.bias3_v = Parameter(torch.FloatTensor(S, num_od))
        Initialization([self.weights1_v, self.weights2_v, self.weights3_v, self.bias1_v, self.bias2_v, self.bias3_v])
        # H^(1)_{s,t}, network 2: on the OD travel times (mean over the 5 paths) of the previous step
        self.weights1_t = Parameter(torch.FloatTensor(S, num_od, hidden_layer[0]))
        self.weights2_t = Parameter(torch.FloatTensor(S, hidden_layer[0], hidden_layer[1]))
        self.weights3_t = Parameter(torch.FloatTensor(S, hidden_layer[1], num_od))
        self.bias1_t = Parameter(torch.FloatTensor(S, hidden_layer[0]))
        self.bias2_t = Parameter(torch.FloatTensor(S, hidden_layer[1]))
        self.bias3_t = Parameter(torch.FloatTensor(S, num_od))
        Initialization([self.weights1_t, self.weights2_t, self.weights3_t, self.bias1_t, self.bias2_t, self.bias3_t])
        # H^(1)_{s,t}: the two outputs (2 x 96 values) joined by one linear layer and tanh -> h_{s,t}
        self.weights_fusion = Parameter(torch.FloatTensor(S, 2 * num_od, num_od))
        self.bias_fusion = Parameter(torch.FloatTensor(S, num_od))
        Initialization([self.weights_fusion, self.bias_fusion])
        # H^(2)_{s,t}: shift term beta_{s,t} of the demand recursion (one linear layer, LeakyReLU)
        self.weights1_v2 = Parameter(torch.FloatTensor(S, num_od, num_od))
        self.bias1_v2 = Parameter(torch.FloatTensor(S, num_od))
        Initialization([self.weights1_v2, self.bias1_v2])
        # attention fusion (Section 4.4)
        self.attention = Attention(num_od, num_link, att_hidden_size)

    def forward(self, demand):
        batch_size = demand.shape[0]
        theta = torch.abs(self.theta)
        link_flow = torch.zeros(S, T, batch_size, num_link)
        demand_pred = torch.zeros(S, T, batch_size, num_od)
        demand_sum = torch.zeros(S, batch_size, num_od)

        for s in range(S):
            # 1. initial demand: H^(0)_s splits the aggregate demand qbar^m over the T steps; q_{s,1} = first step
            d = F.layer_norm(demand, (num_od,))
            d = F.leaky_relu(torch.mm(d, self.weights_d1[s]) + self.bias_d1[s], negative_slope=0.2)
            d = F.leaky_relu(torch.mm(d, self.weights_d2[s]) + self.bias_d2[s], negative_slope=0.2)
            d = F.leaky_relu(torch.mm(d, self.weights_d3[s]) + self.bias_d3[s], negative_slope=0.2)
            d = torch.softmax(d.reshape(batch_size, self.num_od, T), axis=2)           # (batch, num_od, T)
            demand_initial = (d * demand.reshape(batch_size, self.num_od, 1)).permute(2, 0, 1)   # (T, batch, num_od)

            # 2. step 1: Logit loading on delta_s (LP_s) with free-flow path times
            demand_st = demand_initial[0]
            LP_s = LP[s]
            path_time_st = torch.mm(free_flow_tt.repeat(batch_size, 1), LP_s)
            link_flow_st, link_time_st = assignment(demand_st, theta[s, 0], path_time_st, LP_s)
            path_time_st = torch.mm(link_time_st, LP_s)
            od_travel_time = torch.mean(path_time_st.reshape(batch_size, num_od, k), dim=-1)   # mean OD path time

            link_flow_s = torch.zeros(T, batch_size, num_link)
            link_flow_s[0] = link_flow_st
            demand_s = torch.zeros(T, batch_size, num_od)
            demand_s[0] = demand_st
            demand_sum_s = demand_st

            for t in range(T - 1):
                # 3. demand recursion (Section 4.5): q_{s,t+1} = alpha_{s,t} q_{s,t} + beta_{s,t}, alpha_{s,t} = 1 + 0.5 h_{s,t}
                h_v = F.layer_norm(link_flow_st, (num_link,))
                h_v = torch.tanh(torch.mm(h_v, self.weights1_v[s]) + self.bias1_v[s])
                h_v = torch.tanh(torch.mm(h_v, self.weights2_v[s]) + self.bias2_v[s])
                h_v = torch.tanh(torch.mm(h_v, self.weights3_v[s]) + self.bias3_v[s])
                h_t = F.layer_norm(od_travel_time, (num_od,))
                h_t = torch.tanh(torch.mm(h_t, self.weights1_t[s]) + self.bias1_t[s])
                h_t = torch.tanh(torch.mm(h_t, self.weights2_t[s]) + self.bias2_t[s])
                h_t = torch.tanh(torch.mm(h_t, self.weights3_t[s]) + self.bias3_t[s])
                H = torch.tanh(torch.mm(torch.cat([h_v, h_t], dim=1), self.weights_fusion[s]) + self.bias_fusion[s])   # h_{s,t}
                offset = F.leaky_relu(torch.mm(demand_st, self.weights1_v2[s]) + self.bias1_v2[s], negative_slope=0.2)   # beta_{s,t}
                demand_st = (1 + 0.5 * H) * demand_st + offset

                # 2. Logit loading of step t+1 with the BPR path times of step t
                link_flow_st, link_time_st = assignment(demand_st, theta[s, t + 1], path_time_st, LP_s)
                path_time_st = torch.mm(link_time_st, LP_s)
                od_travel_time = torch.mean(path_time_st.reshape(batch_size, num_od, k), dim=-1)

                link_flow_s[t + 1] = link_flow_st
                demand_s[t + 1] = demand_st
                demand_sum_s = demand_sum_s + demand_st

            link_flow[s] = link_flow_s
            demand_pred[s] = demand_s
            demand_sum[s] = demand_sum_s                                               # for the conservation loss L_Q

        # 4. attention fusion (weights lambda^m_{s,t}) of the link flows and the per-step OD demand
        att, link_flow = self.attention(demand, link_flow)                            # att: (T, batch, 1, S)
        demand_pred = torch.matmul(att, demand_pred.permute(1, 2, 0, 3)).reshape(T, batch_size, -1)
        return att.reshape(T, batch_size, S), demand_sum, demand_pred, link_flow


model = MTCG(num_link=num_link, num_od=num_od, hidden_layer=[512, 512], att_hidden_size=128, hidden_layer2=[512, 512])
mse_loss, l1_loss = nn.MSELoss(), nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=BETAS)


def losses(demand_in, flow_obs, idx_ref, out):
    """L_V, L_Q, L_D, L_{Q,T} (code L_q) and the total loss mu_1 L_V + mu_2 L_Q + mu_3 L_D + mu_4 L_{Q,T} (Section 4.6)"""
    att, d_sum, d_pred, v_pred = out
    n = demand_in.shape[0]
    L_V = mse_loss(v_pred[:, :, observed_link - 1], flow_obs[:, :, observed_link - 1])
    L_Q = mse_loss(d_sum, demand_in.reshape(1, n, -1).repeat(S, 1, 1))
    L_D = torch.mean(torch.sum(v_pred * free_flow_tt + 0.03 * (v_pred ** 5) * free_flow_tt / (link_capacity ** 4), dim=-1))
    L_q = mse_loss(d_pred, demand_reference[:T, idx_ref, :])
    return L_V, L_Q, L_D, L_q, w1 * L_V + w2 * L_Q + w3 * L_D + w4 * L_q


loss_train, loss_test = [], []                     # rows: link_flow (L_V), demand (L_Q + L_{Q,T}), vdf (L_D), total
t0 = time.time()
for it in range(n_iters):
    idx = np.random.choice(sample_size, batch_size, replace=False)
    model.zero_grad()
    L_V, L_Q, L_D, L_q, total = losses(train_input[idx], train_output[:, idx, :], idx, model(train_input[idx]))
    total.backward()
    torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1, norm_type=2)
    optimizer.step()
    if args.od_weighted:                            # cosine decay from LR to LR_MIN_FRAC * LR over the iterations
        for g in optimizer.param_groups:
            g['lr'] = LR * (LR_MIN_FRAC + (1 - LR_MIN_FRAC) * 0.5 * (1 + np.cos(np.pi * (it + 1) / n_iters)))
    loss_train.append([L_V.item(), L_Q.item() + L_q.item(), L_D.item(), total.item()])
    if not args.no_test_loss:
        with torch.no_grad():                       # test loss (recorded only, never used for training)
            e = losses(test_input, test_output, slice(NTRAIN, None), model(test_input))
            loss_test.append([e[0].item(), e[1].item() + e[3].item(), e[2].item(), e[4].item()])
    if it % 100 == 0 or it == n_iters - 1:
        print(f'iteration {it + 1}/{n_iters}: L_V {L_V.item():.1f}, L_Q {L_Q.item():.1f}, L_D {L_D.item():.1f}, '
              f'total {total.item():.1f}' + (f', test L_V {loss_test[-1][0]:.1f}' if loss_test else '')
              + f' ({time.time() - t0:.0f} s)', flush=True)

# ---- test errors ----
with torch.no_grad():
    att_pred, _, demand_test_pred, link_flow_test_pred = model(test_input)
demand_test_pred = torch.clamp(demand_test_pred, min=0)                           # reported OD demand >= 0
ref = demand_reference[:, NTRAIN:, :]


def err(obs, est):                                                                  # RMSE, MAE, MAPE: paper Eqs. (42)-(44)
    return [mse_loss(obs, est) ** 0.5, l1_loss(obs, est), torch.mean(torch.abs(obs - est) / obs)]


Error = torch.tensor([err(test_output[:, :, observed_link - 1], link_flow_test_pred[:, :, observed_link - 1]),
                      err(test_output[:, :, unobserved_link - 1], link_flow_test_pred[:, :, unobserved_link - 1]),
                      err(test_output, link_flow_test_pred),
                      err(ref, demand_test_pred)]).numpy()
Error[:, :2] = np.floor(100 * Error[:, :2]) / 100                                 # RMSE, MAE (veh/h): 2 decimals
Error[:, 2] = np.floor(10000 * Error[:, 2]) / 10000                                # MAPE: 4 decimals
Error = pd.DataFrame(Error, index=['With sensors', 'Without sensors', 'Link flow total error', 'OD demand'],
                     columns=['RMSE', 'MAE', 'MAPE'])

# ---- outputs ----
pd.DataFrame({'Estimation': link_flow_test_pred.detach().numpy().reshape(-1),
              'Observation': test_output.numpy().reshape(-1)}).to_excel(os.path.join(output_dir, 'link_flow_estimation.xlsx'), index=False)
pd.DataFrame({'Estimation': demand_test_pred.detach().numpy().reshape(-1),
              'Reference': ref.numpy().reshape(-1)}).to_excel(os.path.join(output_dir, 'od_demand_estimation.xlsx'), index=False)
cols = ['link_flow', 'demand', 'vdf', 'total']
pd.DataFrame(loss_train, columns=cols).to_excel(os.path.join(output_dir, 'loss_train.xlsx'), index=False)
if loss_test:
    pd.DataFrame(loss_test, columns=cols).to_excel(os.path.join(output_dir, 'loss_test.xlsx'), index=False)
Error.to_csv(os.path.join(output_dir, 'error_table.csv'))

with torch.no_grad():
    _, _, _, lf_tr = model(train_input)
o, p = train_output[:, :, observed_link - 1], lf_tr[:, :, observed_link - 1]
th = torch.abs(model.theta).detach()                                               # (S, T, num_od)
w_mean = att_pred.detach().mean(1)                                                 # (T, S) mean test attention
fused = (w_mean.T[:, :, None] * th).sum(0)                                         # attention-weighted theta (T, num_od)
np.savez(os.path.join(output_dir, 'theta_attention.npz'), theta_abs=th.numpy(), attention_test=att_pred.detach().numpy(),
         observed_link=observed_link, unobserved_link=unobserved_link)
info = {'S': S, 'seed': SEED, 'od_weighted': args.od_weighted, 'loss_weights_w1_w2_w3_w4': [w1, w2, w3, w4],
        'adam_lr_betas': [LR, list(BETAS)], 'lr_schedule': 'cosine to 10%' if args.od_weighted else 'constant',
        'iterations': n_iters, 'batch_size': batch_size, 'threads': args.threads, 'seconds': time.time() - T_START,
        'error_table': Error.to_dict(), 'train_sensor_MAPE': float(torch.mean(torch.abs(o - p) / o)),
        'theta_abs_median_all': float(th.median()), 'theta_abs_median_by_template': [float(x) for x in th.reshape(S, -1).median(1).values],
        'theta_fused_median': float(fused.median()), 'theta_fused_q10_q90': [float(fused.quantile(0.1)), float(fused.quantile(0.9))],
        'attention_mean_by_template': [float(x) for x in w_mean.mean(0)],
        'layout': args.layout or 'paper Fig. 13', 'observed_links': [int(x) for x in observed_link]}
json.dump(info, open(os.path.join(output_dir, 'run_info.json'), 'w'), indent=1)
m = Error['MAPE']
print(f'done S={S} seed={SEED}{" OD-weighted" if args.od_weighted else ""} ({info["seconds"]:.0f} s): test MAPE sensors '
      f'{100 * m.iloc[0]:.2f}%, without sensors {100 * m.iloc[1]:.2f}%, total {100 * m.iloc[2]:.2f}%, OD {100 * m.iloc[3]:.2f}% '
      f'| training sensor MAPE {100 * info["train_sensor_MAPE"]:.2f}% -> {output_dir}')
