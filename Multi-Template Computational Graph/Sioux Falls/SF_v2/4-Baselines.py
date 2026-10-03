"""Sioux Falls black-box deep-learning baselines (historical mean, DNN, LSTM, GRU) -> baselines/results/
Models map the day's aggregate OD demand (96 values, sum of the first T = 8 steps) directly to link flows (8 steps x 76 links):
no paths, no route choice, no per-step OD. Compared with the MTCG, S = 1..5.
Same as the MTCG notebook (2-MTCG.ipynb): data files (data/), T = 8, samples 0-399 train /
400-499 test, the 66 sensor links used in the loss and the 10 withheld links, Adam (lr 0.002, betas 0.5 / 0.999),
batches of 64 drawn at random, 1,000 updates, gradient clipping at 1, and the RMSE / MAE / MAPE formulas.
Inputs are standardized with training-day statistics; outputs are scaled with one global mean and standard deviation of
the training-day sensor flows (no information from withheld links or test days).
usage: python 4-Baselines.py [--updates 1000] [--seeds 42,43,44] [--models dnn,lstm,gru] [--tag main]"""
import os, sys, time, json, argparse
import numpy as np, pandas as pd, torch, torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
OUT = os.path.join(HERE, 'baselines', 'results'); os.makedirs(OUT, exist_ok=True)
ap = argparse.ArgumentParser()
ap.add_argument('--updates', type=int, default=1000)
ap.add_argument('--seeds', default='42,43,44')
ap.add_argument('--models', default='dnn,lstm,gru')
ap.add_argument('--tag', default='main')
args = ap.parse_args()

# ---- data: identical to the MTCG notebook, cells 5-21 and 33 ----
T = 8
od_demand = pd.read_csv(os.path.join(DATA, 'demand.csv')).values
num_od = od_demand.shape[1]
od_demand = np.reshape(od_demand, [12, 500, num_od])                  # 12 time steps, 500 days
demand_total = torch.from_numpy(od_demand[:T].sum(axis=0)).float()    # (500, 96) aggregate OD = model input
demand_reference = torch.from_numpy(od_demand[:T]).float()            # (8, 500, 96) only for the OD reference
flow = pd.read_csv(os.path.join(DATA, 'link_flow.csv')).values
num_link = flow.shape[1]
link_flow = torch.from_numpy(np.reshape(flow, [12, -1, num_link])[:T]).float()   # (8, 500, 76)
train_input, test_input = demand_total[:400], demand_total[400:]
train_output, test_output = link_flow[:, :400], link_flow[:, 400:]
unobserved_link = np.array([47, 41, 67, 21, 25, 7, 61, 13, 37, 4])
observed_link = np.array([1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 26, 27, 28,
                          29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52,
                          53, 54, 55, 56, 57, 58, 59, 60, 62, 63, 64, 65, 66, 68, 69, 70, 71, 72, 73, 74, 75, 76])
assert len(observed_link) == 66 and len(np.intersect1d(observed_link, unobserved_link)) == 0 and len(np.union1d(observed_link, unobserved_link)) == num_link
OBS = observed_link - 1

# ---- scaling from training days and sensor links only ----
x_mu, x_sd = train_input.mean(0), train_input.std(0).clamp(min=1e-6)
y_mu, y_sd = float(train_output[:, :, OBS].mean()), float(train_output[:, :, OBS].std())
def x_in(d): return (d - x_mu) / x_sd

class DNN(nn.Module):
    def __init__(self, h=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(num_od, h), nn.LeakyReLU(0.2), nn.Linear(h, h), nn.LeakyReLU(0.2), nn.Linear(h, T * num_link))
    def forward(self, d):                                              # d: (B, 96) -> (T, B, 76)
        return self.net(x_in(d)).reshape(-1, T, num_link).transpose(0, 1) * y_sd + y_mu

class RNN(nn.Module):
    def __init__(self, cell, h=512):
        super().__init__()
        self.rnn = (nn.LSTM if cell == 'lstm' else nn.GRU)(num_od + T, h, num_layers=2, batch_first=True)
        self.head = nn.Linear(h, num_link)
    def forward(self, d):                                              # each step: aggregate OD + one-hot step
        B = d.shape[0]
        x = torch.cat([x_in(d)[:, None, :].expand(B, T, num_od), torch.eye(T)[None].expand(B, T, T)], dim=2)
        y, _ = self.rnn(x)
        return self.head(y).transpose(0, 1) * y_sd + y_mu

def errors(pred):
    """same formulas as the MTCG notebook (cell 47): RMSE = sqrt(MSE), MAE, MAPE = mean(|e| / observed)"""
    rows = {}
    for name, idx in (('With sensors', observed_link - 1), ('Without sensors', unobserved_link - 1), ('Link flow total', np.arange(num_link))):
        o, p = test_output[:, :, idx], pred[:, :, idx]
        rows[name] = (float(((o - p) ** 2).mean().sqrt()), float((o - p).abs().mean()), float(((o - p).abs() / o).mean()))
    return rows

def train(model_name, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = DNN() if model_name == 'dnn' else RNN(model_name)
    opt = torch.optim.Adam(model.parameters(), lr=0.002, betas=(0.5, 0.999))
    t0 = time.time(); hist = []
    for it in range(args.updates):
        idx = np.random.choice(400, 64, replace=False)
        pred = model(train_input[idx])
        loss = nn.functional.mse_loss(pred[:, :, OBS], train_output[:, idx][:, :, OBS])   # sensor links only
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
        opt.step(); hist.append(loss.item())
    with torch.no_grad():
        pred_te = model(test_input); pred_tr = model(train_input)
    tr = float((((train_output[:, :, OBS] - pred_tr[:, :, OBS]).abs()) / train_output[:, :, OBS]).mean())
    return errors(pred_te), tr, time.time() - t0, hist

rows = []
# historical mean: per step and sensor link, mean over training days; withheld links get the per-step mean of all sensor links
hm = train_output.mean(1)                                              # (T, 76)
hm[:, unobserved_link - 1] = train_output[:, :, OBS].mean((1, 2))[:, None]
e = errors(hm[:, None, :].expand(T, 100, num_link))
rows += [{'model': 'Historical mean', 'seed': '-', 'type': k, 'RMSE': v[0], 'MAE': v[1], 'MAPE': v[2]} for k, v in e.items()]
# OD reference for models without OD output: aggregate demand split evenly over the T steps
od_even = (test_input / T)[None].expand(T, 100, num_od); od_ref = demand_reference[:, 400:]
od_e = (float(((od_ref - od_even) ** 2).mean().sqrt()), float((od_ref - od_even).abs().mean()), float(((od_ref - od_even).abs() / od_ref).mean()))
rows.append({'model': 'Even split of aggregate OD', 'seed': '-', 'type': 'OD demand', 'RMSE': od_e[0], 'MAE': od_e[1], 'MAPE': od_e[2]})
info = {}
for m in args.models.split(','):
    for s in [int(x) for x in args.seeds.split(',')]:
        e, tr_mape, sec, hist = train(m, s)
        rows += [{'model': m.upper(), 'seed': s, 'type': k, 'RMSE': v[0], 'MAE': v[1], 'MAPE': v[2]} for k, v in e.items()]
        info[f'{m}_{s}'] = {'train_sensor_MAPE': tr_mape, 'seconds': sec, 'loss_first': hist[0], 'loss_last50': float(np.mean(hist[-50:]))}
        print(f'{m.upper():5s} seed {s}: test sensors MAPE {e["With sensors"][2]:.4f}, withheld {e["Without sensors"][2]:.4f}, total {e["Link flow total"][2]:.4f}; '
              f'train sensors MAPE {tr_mape:.4f}; {sec:.0f} s', flush=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, f'runs_{args.tag}.csv'), index=False)
json.dump({'updates': args.updates, 'y_mu': y_mu, 'y_sd': y_sd, 'runs': info}, open(os.path.join(OUT, f'info_{args.tag}.json'), 'w'), indent=1)
summ = df.groupby(['model', 'type'])[['RMSE', 'MAE', 'MAPE']].agg(['mean', 'std'])
summ.to_csv(os.path.join(OUT, f'summary_{args.tag}.csv'))
print(summ.round(4).to_string())
