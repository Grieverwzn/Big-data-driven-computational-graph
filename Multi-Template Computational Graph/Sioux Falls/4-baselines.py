"""
Sioux Falls baseline models for the comparison with the Multi-Template Computational Graph (MTCG), paper "Can
Contextual Archetypes Explain Daily Traffic Variation? A Multi-Template Computational Graph with Attention-Based
Fusion" (Section 7.2.3, Table 6; error formulas Eqs. (42)-(44)).

Authors:
    Xin (Bruce) Wu, Department of Civil and Environmental Engineering, Villanova University, PA, USA
    Feng Shao, School of Mathematics, China University of Mining and Technology, China

Contact: xwu03@villanova.edu (Villanova University), xinwu8592@gmail.com (personal)

MIT License
Copyright (c) 2026 Xin (Bruce) Wu, Feng Shao

Usage  : python "4-baselines.py"                               (all models, seeds 42, 43, 44)
         python "4-baselines.py" --models dnnpath,gcn --seeds 42 (a subset; rows of other models already in
                                                                  results/baselines/runs.csv are kept)
         options: --models (dnn, lstm, gru, dnnpath, gcn), --seeds, --updates (default 1000), --threads (default 4)
         Needs data/ (1-data_generation.py) and sf_common.py next to this script (numpy, pandas, torch). CPU; all
         models and seeds take about 30-60 min (LSTM and GRU are the slowest).

All baselines work under the same conditions as the MTCG: the same input (the aggregate OD demand of a sample, 96
values = sum of the 8 steps 7:00-8:45), the same supervision (link flows of the 66 links with sensors on the 400
training samples), the same 100 test samples and the same error formulas.
  Historical mean   per step and link with sensor, mean over the training samples (no randomness)    output: S
  DNN, LSTM, GRU    black box: aggregate OD -> link flows (8 x 76); the links without sensors are never in the loss,
                    so only the links with sensors are reported                                       output: S
  DNN-path          network structure, no behaviour model: a DNN gives each OD pair's split over its 5 free-flow
                    shortest paths at each step (softmax); path flows = split x (aggregate OD / 8); link flows through
                    the link-path incidence matrix                                                    output: S, W, OD
  GCN               link graph (76 nodes; an edge when one link ends where the other starts); node input = the link's
                    demand exposure (aggregate OD / 8 spread evenly over each OD pair's 5 free-flow paths, kept apart by
                    path rank: 5 values) and its free-flow time; 3 graph-convolution layers of 64 with residual
                    connections; output 8 steps per link                                             output: S, W
  (S = links with sensors, W = links without sensors, OD = per-step OD demand; DNN-path's OD output is the even split of
  the aggregate demand, it is listed but not used in the paper.)
Training: Adam (betas 0.9 / 0.999), batches of 64, 1000 updates, gradient clipping at norm 1 (as the MTCG); loss = mean
squared error on the links with sensors (scaled by the standard deviation of the training flows). The learning rate is
chosen from {0.001, 0.002, 0.005} by the sensor MAPE on validation samples 350-399 when trained on samples 0-349 (seed
42); then each seed is trained on samples 0-399 with that rate.

Output (results/baselines/): runs.csv (model, seed, error type, RMSE, MAE, MAPE; one row per output a model can give),
info.json (chosen learning rates with the validation MAPEs, run times, training-sample sensor MAPE), summary_mean_sd.csv
(mean and standard deviation over the seeds, MAPE in %).
"""
import os, time, json, argparse
import numpy as np, pandas as pd, torch, torch.nn as nn
from sf_common import DATA, RESULTS, K, T, NSTEP, NDAY, NTRAIN, UNOBSERVED, link_attributes, load_data

OUT = os.path.join(RESULTS, 'baselines'); os.makedirs(OUT, exist_ok=True)
ap = argparse.ArgumentParser(description='Sioux Falls baselines')
ap.add_argument('--models', default='dnn,lstm,gru,dnnpath,gcn')
ap.add_argument('--seeds', default='42,43,44')
ap.add_argument('--updates', type=int, default=1000)
ap.add_argument('--threads', type=int, default=4)
args = ap.parse_args(); torch.set_num_threads(args.threads)
LRS, SEEDS = (0.001, 0.002, 0.005), tuple(int(s) for s in args.seeds.split(','))

# ---- data (as the MTCG) ----
od, flow = load_data(); num_od, num_link = od.shape[2], flow.shape[2]
agg = torch.from_numpy(od[:T].sum(0)).float()                                   # (500, 96) aggregate OD demand
od_ref = torch.from_numpy(od[:T, NTRAIN:]).float()                              # (8, 100, 96) per-step OD of the test samples
V = torch.from_numpy(flow[:T]).float()                                          # (8, 500, 76)
UNOBS = UNOBSERVED - 1
OBS = np.setdiff1d(np.arange(num_link), UNOBS); assert len(OBS) == 66
LP = torch.from_numpy(np.load(os.path.join(DATA, 'paths_template1.npz'))['LP']).float()   # (76, 480) free-flow path set
la = link_attributes
fft = torch.tensor(la['free flow travel time'].values, dtype=torch.float32)


def errors(pred_v, pred_od=None, days=slice(NTRAIN, NDAY), with_w=True):
    """test RMSE, MAE, MAPE of the link flows (with sensors; without sensors and all links if with_w) and the OD demand"""
    o = V[:, days]; r = {}
    sets = [('With sensors', OBS)] + ([('Without sensors', UNOBS), ('Link flow total', np.arange(num_link))] if with_w else [])
    for name, idx in sets:
        a, p = o[:, :, idx], pred_v[:, :, idx]
        r[name] = (float(((a - p) ** 2).mean().sqrt()), float((a - p).abs().mean()), float(((a - p).abs() / a).mean()))
    if pred_od is not None:
        r['OD demand'] = (float(((od_ref - pred_od) ** 2).mean().sqrt()), float((od_ref - pred_od).abs().mean()), float(((od_ref - pred_od).abs() / od_ref).mean()))
    return r


# ---- models ----
class Scaled(nn.Module):
    """input standardization and output scaling from the given training samples and links with sensors only"""
    def set_scale(self, days):
        self.x_mu, self.x_sd = agg[days].mean(0), agg[days].std(0).clamp(min=1e-6)
        self.y_mu, self.y_sd = float(V[:, days][:, :, OBS].mean()), float(V[:, days][:, :, OBS].std())

    def x_in(self, d): return (d - self.x_mu) / self.x_sd


class DNN(Scaled):
    def __init__(self, h=512):
        super().__init__(); self.net = nn.Sequential(nn.Linear(num_od, h), nn.LeakyReLU(0.2), nn.Linear(h, h), nn.LeakyReLU(0.2), nn.Linear(h, T * num_link))

    def forward(self, d): return self.net(self.x_in(d)).reshape(-1, T, num_link).transpose(0, 1) * self.y_sd + self.y_mu, None


class RNN(Scaled):
    """two-layer LSTM or GRU over the 8 steps; input at each step = standardized aggregate OD + one-hot step"""
    def __init__(self, cell, h=512):
        super().__init__(); self.rnn = (nn.LSTM if cell == 'lstm' else nn.GRU)(num_od + T, h, num_layers=2, batch_first=True); self.head = nn.Linear(h, num_link)

    def forward(self, d):
        B = d.shape[0]
        x = torch.cat([self.x_in(d)[:, None, :].expand(B, T, num_od), torch.eye(T)[None].expand(B, T, T)], dim=2)
        return self.head(self.rnn(x)[0]).transpose(0, 1) * self.y_sd + self.y_mu, None


class DNNPath(Scaled):
    def __init__(self, h=512):
        super().__init__(); self.net = nn.Sequential(nn.Linear(num_od, h), nn.LeakyReLU(0.2), nn.Linear(h, h), nn.LeakyReLU(0.2), nn.Linear(h, T * num_od * K))

    def forward(self, d):
        B = d.shape[0]
        split = torch.softmax(self.net(self.x_in(d)).reshape(B, T, num_od, K), dim=3)
        q_step = (d / T)[:, None, :].expand(B, T, num_od)                       # even split of the aggregate OD
        f = (split * q_step[..., None]).reshape(B, T, num_od * K)               # path flows
        return (f @ LP.T).transpose(0, 1), q_step.transpose(0, 1)               # (T, B, 76), (T, B, 96)


# link graph: edge between links a and b when a ends where b starts (either direction), self loops, symmetric normalization
ends = la[['start', 'end']].values
A = torch.zeros(num_link, num_link)
for a in range(num_link):
    for b in range(num_link):
        if a != b and (ends[a, 1] == ends[b, 0] or ends[b, 1] == ends[a, 0]): A[a, b] = 1
A = A + torch.eye(num_link); dinv = A.sum(1) ** -0.5; A_hat = dinv[:, None] * A * dinv[None]


def exposure(d):
    """(B, 96) -> (B, 76, K): aggregate OD / 8 spread evenly over the 5 paths, kept apart by path rank (1st..5th)"""
    q = (d / T).repeat_interleave(K, dim=1) / K
    return torch.stack([(q * (torch.arange(num_od * K) % K == r).float()) @ LP.T for r in range(K)], dim=2)


class GCN(Scaled):
    """graph convolution on the link graph with residual connections; node input: the demand exposure by path rank
    (5 values) and the free-flow time"""
    def __init__(self, h=64, layers=3):
        super().__init__(); self.inp = nn.Linear(K + 1, h); self.W = nn.ModuleList([nn.Linear(h, h) for _ in range(layers)]); self.head = nn.Linear(h, T)

    def set_scale(self, days):
        super().set_scale(days); e = exposure(agg[days]); self.e_mu, self.e_sd = e.mean((0, 1)), e.std((0, 1)).clamp(min=1e-6)

    def forward(self, d):
        B = d.shape[0]
        x = torch.cat([(exposure(d) - self.e_mu) / self.e_sd, ((fft - fft.mean()) / fft.std())[None, :, None].expand(B, -1, 1)], dim=2)
        x = self.inp(x)
        for W in self.W:
            x = x + nn.functional.leaky_relu(A_hat @ W(x), 0.2)
        return self.head(x).permute(2, 0, 1) * self.y_sd + self.y_mu, None      # (T, B, 76)


MAKE = {'dnn': DNN, 'lstm': lambda: RNN('lstm'), 'gru': lambda: RNN('gru'), 'dnnpath': DNNPath, 'gcn': GCN}
NAME = {'dnn': 'DNN', 'lstm': 'LSTM', 'gru': 'GRU', 'dnnpath': 'DNN-path', 'gcn': 'GCN'}
GIVES_W = {'dnn': False, 'lstm': False, 'gru': False, 'dnnpath': True, 'gcn': True}


def fit(m, seed, lr, tr_days):
    torch.manual_seed(seed); np.random.seed(seed)
    model = MAKE[m](); model.set_scale(tr_days)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999)); y_sd = model.y_sd
    for it in range(args.updates):
        idx = np.random.choice(tr_days, 64, replace=False)
        pv, _ = model(agg[idx])
        loss = nn.functional.mse_loss(pv[:, :, OBS] / y_sd, V[:, idx][:, :, OBS] / y_sd)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2); opt.step()
    return model


def sensor_mape(model, days):
    with torch.no_grad():
        pv, _ = model(agg[days])
    o = V[:, days][:, :, OBS]; return float(((o - pv[:, :, OBS]).abs() / o).mean())


# earlier results of models not rerun now are kept
models = [x for x in args.models.split(',') if x]
p_runs, p_info = os.path.join(OUT, 'runs.csv'), os.path.join(OUT, 'info.json')
old = pd.read_csv(p_runs) if os.path.exists(p_runs) else pd.DataFrame(columns=['model', 'seed', 'type', 'RMSE', 'MAE', 'MAPE'])
old = old[~old.model.isin(['Historical mean'] + [NAME[m] for m in models])]
info = json.load(open(p_info)) if os.path.exists(p_info) else {'learning_rate': {}, 'seconds': {}, 'train_sensor_MAPE': {}}
rows = []


def add(model_name, seed, e):
    rows.extend({'model': model_name, 'seed': seed, 'type': k, 'RMSE': v[0], 'MAE': v[1], 'MAPE': v[2]} for k, v in e.items())


def save():
    df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(p_runs, index=False); json.dump(info, open(p_info, 'w'), indent=1)
    return df


# historical mean (links with sensors only)
hm = V[:, :NTRAIN].mean(1)[:, None, :].expand(T, NDAY - NTRAIN, num_link)
add('Historical mean', '-', errors(hm, with_w=False))

for m in models:
    val = {}
    for lr in LRS:                                     # learning rate: validation samples 350-399, seed 42
        t0 = time.time(); model = fit(m, 42, lr, np.arange(350)); val[lr] = sensor_mape(model, np.arange(350, 400))
        print(f'{NAME[m]} lr {lr}: validation sensor MAPE {val[lr]:.4f} ({time.time() - t0:.0f} s)', flush=True)
    lr = min(val, key=val.get); info['learning_rate'][NAME[m]] = {'chosen': lr, 'validation sensor MAPE': val}
    for s in SEEDS:
        t0 = time.time(); model = fit(m, s, lr, np.arange(NTRAIN))
        with torch.no_grad():
            pv, pod = model(agg[NTRAIN:])
        e = errors(pv, pod, with_w=GIVES_W[m]); add(NAME[m], s, e)
        info['seconds'][f'{NAME[m]}_{s}'] = time.time() - t0; info['train_sensor_MAPE'][f'{NAME[m]}_{s}'] = sensor_mape(model, np.arange(NTRAIN))
        print(f'{NAME[m]} seed {s} (lr {lr}): test MAPE ' + ', '.join(f'{k} {v[2]:.4f}' for k, v in e.items()), flush=True)
    save()                                             # after each model (restart-safe record)

df = save()
df['MAPE'] = 100 * df['MAPE']
summ = df.groupby(['model', 'type'], sort=False).agg(n=('RMSE', 'size'), RMSE_mean=('RMSE', 'mean'), RMSE_sd=('RMSE', 'std'),
                                                     MAE_mean=('MAE', 'mean'), MAE_sd=('MAE', 'std'),
                                                     MAPE_mean=('MAPE', 'mean'), MAPE_sd=('MAPE', 'std')).reset_index()
summ.to_csv(os.path.join(OUT, 'summary_mean_sd.csv'), index=False)
print(summ.round(2).to_string(index=False))
