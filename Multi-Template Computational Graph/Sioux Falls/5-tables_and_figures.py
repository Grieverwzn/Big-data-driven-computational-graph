"""
Sioux Falls tables and figures of the Multi-Template Computational Graph (MTCG), paper "Can Contextual Archetypes
Explain Daily Traffic Variation? A Multi-Template Computational Graph with Attention-Based Fusion" (Section 7.2:
Tables 5-7 and Figs. 14-19).

Authors:
    Xin (Bruce) Wu, Department of Civil and Environmental Engineering, Villanova University, PA, USA
    Feng Shao, School of Mathematics, China University of Mining and Technology, China

Contact: xwu03@villanova.edu (Villanova University), xinwu8592@gmail.com (personal)

MIT License
Copyright (c) 2026 Xin (Bruce) Wu, Feng Shao

Usage  : python "5-tables_and_figures.py"
         Needs data/ (1-data_generation.py), the training runs in results/ (3-mtcg.py), the baselines in
         results/baselines/ (4-baselines.py), the coverage runs in results/coverage/ (3-mtcg.py --layout), and
         sf_common.py next to this script (numpy, pandas, matplotlib, networkx, pillow, openpyxl). All paths are
         relative to this script. Uses whatever runs exist: a table row is the mean +- standard deviation over the
         runs found, and a table or figure whose runs are missing is skipped with a message. About 1-2 minutes.

Inputs : data/demand.csv, link_flow.csv, od_pair.csv, link_attributes.csv, day_info.csv
         results/S<S>_seed<seed>[_odw]/run_info.json (all runs; tables) and, for the runs drawn in the figures,
         link_flow_estimation.xlsx, od_demand_estimation.xlsx, loss_train.xlsx, loss_test.xlsx, theta_attention.npz
         results/baselines/runs.csv, results/coverage/<layout>_s<seed>_S<S>/run_info.json

Outputs (paper numbers as in the current manuscript):
  tables/table5.csv, table5.tex            Table 5  mean OD demand and congestion level by time step        Sec. 7.2.1
  tables/table6_mean_sd.csv, table6.tex    Table 6  baselines, MTCG S = 1..5, MTCG S = 5 (OD-weighted):
                                                    test RMSE / MAE / MAPE (%), mean +- sd over the seeds     Sec. 7.2.3
  tables/table6_runs.csv                            every MTCG run of Table 6
  tables/table7_mean_sd.csv, table7.tex    Table 7  sensor coverage: mean +- sd over 5 random layouts       Sec. 7.2.5
  tables/table7_runs.csv                            every coverage run, with the run kept per layout and S
  figures/fig_sf_od_nodes.png/.pdf         Fig. 14  trips leaving / arriving at each node (data only)       Sec. 7.2.1
  figures/fig_sf_svd.png/.pdf              Fig. 15  stacked SVD of the training link flows, N_sigma         Sec. 7.2.2
  figures/fig_time_dynamics_s5_v2.png/.jpg Fig. 16  estimates over the time steps, S = 5                    Sec. 7.2.3
  figures/fig_sf_link_error_D.png/.jpg     Fig. 17  per-link MAPE (S = 1, S = 5), MAPE and attention by day type  Sec. 7.2.4
  figures/fig_scatter_s1_new.png/.jpg,
          fig_scatter_s5_new.png/.jpg      Fig. 18  estimated vs observed link flows and OD trips, (a) S = 1, (b) S = 5  Sec. 7.2.4
  figures/fig_sf_convergence_od_weighted.png/.jpg  Fig. 19  training and test losses, OD-weighted S = 5     Sec. 7.2.4
The figure code is the code of the paper figures (same data, computations, panel order, colours and fonts). For the
figures of one setting (S = 1, S = 5, S = 5 OD-weighted), the run with the lowest training-sample sensor MAPE among
the seeds is used (no test data are used to choose it): S = 1 seed 43, S = 5 seed 44, S = 5 OD-weighted seed 43.
"""
import os, json, glob
import numpy as np, pandas as pd, networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from PIL import Image
from sf_common import HERE, DATA, RESULTS, T, NSTEP, NDAY, NTRAIN, UNOBSERVED, num_link, link_attributes, fft, cap, bpr, run_dir

TAB, FIG = os.path.join(HERE, 'tables'), os.path.join(HERE, 'figures')
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
TYPES = ['Link flow (with sensors)', 'Link flow (without sensors)', 'Link flow (total)', 'OD demand']   # rows of Table 6
KEYS = dict(zip(TYPES, ['With sensors', 'Without sensors', 'Link flow total error', 'OD demand']))   # keys in run_info.json
BKEYS = {'With sensors': TYPES[0], 'Without sensors': TYPES[1], 'Link flow total': TYPES[2], 'OD demand': TYPES[3]}   # baselines
METRICS = ('RMSE', 'MAE', 'MAPE')               # paper Eqs. (rmse)-(mape); RMSE and MAE in veh/h, MAPE in %
BASE = ['Historical mean', 'DNN', 'LSTM', 'GRU', 'DNN-path', 'GCN']


def pm(m, s):
    return f'{m:.2f}' if pd.isna(s) else f'{m:.2f} ± {s:.2f}'


# =====================================================================================================================
# Table 5 (Sec. 7.2.1): designed and generated mean OD demand, v/c and travel-time ratio by time step
# =====================================================================================================================
def table5():
    """designed qbar_t (veh/h, see 1-data_generation.py), mean generated demand q^m_{w,t} over all OD pairs and the 500
    samples, mean v_a / c_a and t_a(v_a) / t_a^0 over all links and samples, steps t = 1..8 (7:00-8:45)"""
    q = pd.read_csv(os.path.join(DATA, 'demand.csv')).values.reshape(NSTEP, NDAY, -1)[:T]
    v = pd.read_csv(os.path.join(DATA, 'link_flow.csv')).values.reshape(NSTEP, NDAY, -1)[:T]
    qbar = 0.3 * (832.0 + np.array([0, 60, 120, 180, 240, 300, 270, 210]))        # QBAR of 1-data_generation.py x 0.3
    t = pd.DataFrame({'time step t': np.arange(1, T + 1), 'start time': [f'{7 + (15 * i) // 60}:{(15 * i) % 60:02d}' for i in range(T)],
                      'designed qbar_t (veh/h)': qbar, 'generated mean (veh/h)': q.mean((1, 2)),
                      'v_a/c_a': (v / cap).mean((1, 2)), 't_a(v_a)/t_a^0': np.stack([bpr(v[s]) / fft for s in range(T)]).mean((1, 2))})
    t.to_csv(os.path.join(TAB, 'table5.csv'), index=False)
    row = lambda name, x, f: name + ' & ' + ' & '.join(f.format(a) for a in x) + r' \\'
    L = [row('Time step $t$', t['time step t'], '{}'), row('Start time', t['start time'], '{}'), r'\midrule',
         row(r'Designed $\bar{q}_{t}$ (veh/h)', t['designed qbar_t (veh/h)'], '{:.0f}'),
         row('Generated mean (veh/h)', t['generated mean (veh/h)'], '{:.0f}'),
         row('$v_a/c_a$', t['v_a/c_a'], '{:.2f}'), row('$t_a(v_a)/t_a^0$', t['t_a(v_a)/t_a^0'], '{:.2f}'), r'\bottomrule']
    open(os.path.join(TAB, 'table5.tex'), 'w').write('\n'.join(L) + '\n')
    print('table 5: written tables/table5.csv, table5.tex'); print(t.round(2).to_string(index=False))


# =====================================================================================================================
# Table 6 (Sec. 7.2.3): baselines and MTCG, mean +- sd over the seeds
# =====================================================================================================================
def runs_of(S, odw):
    """run_info of every finished seed of one setting (S templates; odw: OD-weighted setting, mu_2 = 5 and lr decay)"""
    out = {}
    for p in sorted(glob.glob(os.path.join(RESULTS, f'S{S}_seed*' + ('_odw' if odw else ''), 'run_info.json'))):
        d = json.load(open(p))
        if d['S'] == S and d['od_weighted'] == odw:
            out[d['seed']] = d
    return out


def best_run(S, odw=False):
    """folder of the run with the lowest training-sample sensor MAPE (None if there is no run)"""
    r = runs_of(S, odw)
    if not r:
        return None
    seed = min(r, key=lambda s: r[s]['train_sensor_MAPE'])
    return run_dir(S, seed, odw)


def table6():
    rows = []
    settings = [(f'MTCG S={S}', S, False) for S in range(1, 6)] + [('MTCG S=5 (OD-weighted)', 5, True)]
    for name, S, odw in settings:
        for seed, d in runs_of(S, odw).items():
            e = d['error_table']
            for ty in TYPES:
                rows.append({'model': name, 'seed': seed, 'type': ty, 'RMSE': e['RMSE'][KEYS[ty]], 'MAE': e['MAE'][KEYS[ty]],
                             'MAPE': 100 * e['MAPE'][KEYS[ty]], 'train_sensor_MAPE': 100 * d['train_sensor_MAPE']})
    runs = pd.DataFrame(rows)
    if len(runs):
        runs.to_csv(os.path.join(TAB, 'table6_runs.csv'), index=False)
    pb = os.path.join(RESULTS, 'baselines', 'runs.csv')
    if os.path.exists(pb):
        b = pd.read_csv(pb); b = b[b.model.isin(BASE) & (b.type != 'OD demand')].copy()   # baselines: link flows only
        b['MAPE'] = 100 * b['MAPE']; b['type'] = b['type'].map(BKEYS)
    else:
        print('table 6: results/baselines/runs.csv not found, baselines left out'); b = pd.DataFrame(columns=['model', 'seed', 'type'] + list(METRICS))
    allr = pd.concat([b[['model', 'seed', 'type'] + list(METRICS)], runs[['model', 'seed', 'type'] + list(METRICS)] if len(runs) else None], ignore_index=True)
    order = BASE + [n for n, _, _ in settings]
    t = []
    for mod in order:
        for ty in TYPES:
            g = allr[(allr.model == mod) & (allr.type == ty)]
            if not len(g):
                continue
            r = {'model': mod, 'type': ty, 'n_seeds': len(g) if g.seed.astype(str).ne('-').all() else 0,
                 'seeds': '/'.join(g.seed.astype(str))}
            for k in METRICS:
                r[f'{k}_mean'] = g[k].mean(); r[f'{k}_sd'] = g[k].std() if len(g) > 1 else np.nan; r[k] = pm(r[f'{k}_mean'], r[f'{k}_sd'])
            t.append(r)
    t = pd.DataFrame(t)
    if not len(t):
        print('table 6: no results found'); return
    t.to_csv(os.path.join(TAB, 'table6_mean_sd.csv'), index=False, encoding='utf-8-sig')

    # LaTeX body: one block per model, best mean of each flow type and metric bold and underlined
    best = {(ty, k): t[t.type == ty][f'{k}_mean'].min() for ty in TYPES for k in METRICS if (t.type == ty).any()}
    def cell(r, k):
        v = f'{r[f"{k}_mean"]:.2f}' + ('' if pd.isna(r[f'{k}_sd']) else rf' $\pm$ {r[f"{k}_sd"]:.2f}')
        return rf'\underline{{\textbf{{{v}}}}}' if np.isclose(r[f'{k}_mean'], best[(r.type, k)]) else v
    label = {f'MTCG S={S}': rf'MTCG $S\!=\!{S}$' for S in range(1, 6)}
    label['MTCG S=5 (OD-weighted)'] = r'\makecell[l]{MTCG $S\!=\!5$\\(OD-weighted)}'
    L = []
    for mod in order:
        g = t[t.model == mod]
        if not len(g):
            continue
        L.append(rf'\multirow{{{len(g)}}}{{*}}{{{label.get(mod, mod)}}}' if len(g) > 1 else label.get(mod, mod))
        for _, r in g.iterrows():
            L.append(f' & {r.type} & ' + ' & '.join(cell(r, k) for k in METRICS) + r' \\')
        L.append(r'\midrule')
    L[-1] = r'\bottomrule'
    open(os.path.join(TAB, 'table6.tex'), 'w').write('\n'.join(L) + '\n')
    print('table 6: written tables/table6_mean_sd.csv, table6_runs.csv, table6.tex')
    print(t[['model', 'type', 'seeds', 'RMSE', 'MAE', 'MAPE']].to_string(index=False))


# =====================================================================================================================
# Table 7 (Sec. 7.2.5): sensor coverage, S = 1 vs S = 5, mean +- sd over 5 random layouts per coverage level
# =====================================================================================================================
def table7():
    """for each layout (data/sensor_layouts.csv) and S, the seed with the lowest training-sample sensor MAPE is kept;
    reported: test RMSE and MAPE of all links and of the links without sensors under that layout"""
    rows = []
    for p in sorted(glob.glob(os.path.join(RESULTS, 'coverage', '*', 'run_info.json'))):
        d = json.load(open(p)); e = d['error_table']
        rows.append({'layout': d['layout'], 'n_observed': len(d['observed_links']), 'S': d['S'], 'seed': d['seed'],
                     'train_sensor_MAPE': 100 * d['train_sensor_MAPE'],
                     'total_RMSE': e['RMSE']['Link flow total error'], 'total_MAPE': 100 * e['MAPE']['Link flow total error'],
                     'without_RMSE': e['RMSE']['Without sensors'], 'without_MAPE': 100 * e['MAPE']['Without sensors']})
    if not rows:
        print('table 7: no coverage runs in results/coverage/, skipped'); return
    runs = pd.DataFrame(rows).sort_values(['n_observed', 'layout', 'S', 'seed']).reset_index(drop=True)
    runs['kept'] = False
    runs.loc[runs.groupby(['layout', 'S']).train_sensor_MAPE.idxmin(), 'kept'] = True
    runs.to_csv(os.path.join(TAB, 'table7_runs.csv'), index=False)
    M = ['total_RMSE', 'total_MAPE', 'without_RMSE', 'without_MAPE']
    t = runs[runs.kept].groupby(['n_observed', 'S'])[M].agg(['mean', 'std', 'size'])
    out = pd.DataFrame({'coverage': [f'{100 * n / num_link:.1f}% ({n}/{num_link})' for n, _ in t.index],
                        'S': [S for _, S in t.index], 'layouts': t[(M[0], 'size')].values})
    for m in M:
        out[f'{m}_mean'] = t[(m, 'mean')].values; out[f'{m}_sd'] = t[(m, 'std')].values
    out.to_csv(os.path.join(TAB, 'table7_mean_sd.csv'), index=False)
    L = []
    for cov, g in out.groupby('coverage', sort=False):          # rows S = 1 and S = 5; the better value of each pair bold
        for _, r in g.iterrows():
            cells = []
            for m in M:
                v = f'{r[f"{m}_mean"]:.2f}$\\pm${r[f"{m}_sd"]:.2f}'
                cells.append(rf'\textbf{{{v}}}' if len(g) == 2 and r[f'{m}_mean'] == g[f'{m}_mean'].min() else v)
            L.append(cov.replace('%', r'\%') + f' & {r.S} & ' + ' & '.join(cells) + r' \\')
        L.append(r'\midrule')
    L[-1] = r'\bottomrule'
    open(os.path.join(TAB, 'table7.tex'), 'w').write('\n'.join(L) + '\n')
    print(f'table 7: written tables/table7_mean_sd.csv, table7_runs.csv, table7.tex ({len(runs)} runs)')
    print(out.round(2).to_string(index=False))


# =====================================================================================================================
# Figures (code of the paper figure scripts; only the data paths differ)
# =====================================================================================================================
def save(fig, name, exts=('png', 'jpg'), dpi=300, quality=95):
    """PNG at the given dpi; the JPG used in the paper is converted from the PNG (PIL, given quality)"""
    p = os.path.join(FIG, f'{name}.png')
    fig.savefig(p, dpi=dpi, bbox_inches='tight')
    if 'pdf' in exts:
        fig.savefig(p[:-4] + '.pdf', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    if 'jpg' in exts:
        Image.open(p).convert('RGB').save(p[:-4] + '.jpg', quality=quality)
    print('figure:', name)


def serif(size=None):
    """matplotlib defaults + Times New Roman (the settings of the paper figure scripts)"""
    plt.rcdefaults()
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'mathtext.fontset': 'stix'})
    if size:
        plt.rcParams['font.size'] = size


def read_estimates(d):
    """test estimates of one run: link flows (Estimation, Observation) and OD demand (Estimation, Reference)"""
    lf = pd.read_excel(os.path.join(d, 'link_flow_estimation.xlsx')); od = pd.read_excel(os.path.join(d, 'od_demand_estimation.xlsx'))
    return lf, od


def fig_od_nodes():
    """Fig. 14 (Sec. 7.2.1): for each node, box plots over the 500 samples of the trips leaving / arriving at the node,
    7:00-9:00 (sum over the 8 steps of the hourly rates q^m_{w,t} x 0.25 h)"""
    serif(13)
    q = pd.read_csv(os.path.join(DATA, 'demand.csv')).values.reshape(NSTEP, NDAY, -1)[:T]
    od = pd.read_csv(os.path.join(DATA, 'od_pair.csv')).values
    trips = q.sum(0) * 0.25                                                      # (500, 96) vehicles, 7:00-9:00
    nodes = np.arange(1, 25)
    O = [trips[:, od[:, 0] == n].sum(1) for n in nodes]
    Dd = [trips[:, od[:, 1] == n].sum(1) for n in nodes]
    fig, ax = plt.subplots(figsize=(13, 4.6))
    style = dict(widths=0.32, patch_artist=True, showfliers=True, medianprops=dict(color='black', lw=1.2),
                 flierprops=dict(marker='.', markersize=3, alpha=0.5))
    b1 = ax.boxplot(O, positions=nodes - 0.19, **style)
    b2 = ax.boxplot(Dd, positions=nodes + 0.19, **style)
    for b, col in ((b1, '#4C72B0'), (b2, '#DD8452')):
        for p in b['boxes']:
            p.set_facecolor(col); p.set_alpha(0.8)
    ax.set_xticks(nodes); ax.set_xticklabels([str(n) for n in nodes]); ax.set_xlim(0.4, 24.6)
    ax.set_xlabel('Node ID'); ax.set_ylabel('Number of trips, 7:00–9:00 AM (veh)')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(handles=[Patch(facecolor='#4C72B0', alpha=0.8, label='Leaving the node (origin)'),
                       Patch(facecolor='#DD8452', alpha=0.8, label='Arriving at the node (destination)')],
              loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    save(fig, 'fig_sf_od_nodes', exts=('png', 'pdf'))


def fig_svd():
    """Fig. 15 (Sec. 7.2.2): stacked centred SVD of the observed link flows of the training samples (400 samples x 8 steps
    x 66 links with sensors; mean of each step removed; 3200 x 66 matrix): (a) share of sigma_k^2, (b) ratio
    sigma_k^2 / sigma_{k+1}^2, whose maximum gives N_sigma (paper Section 5.1)"""
    serif(14)
    v = pd.read_csv(os.path.join(DATA, 'link_flow.csv')).values.reshape(NSTEP, NDAY, -1)[:T, :NTRAIN]
    x = v[:, :, np.setdiff1d(np.arange(v.shape[2]), UNOBSERVED - 1)]           # (8, 400, 66)
    x = x - x.mean(1, keepdims=True)                                            # remove the mean of each time step
    s2 = np.linalg.svd(x.reshape(-1, x.shape[2]), compute_uv=False) ** 2        # squared singular values sigma_k^2
    share = s2 / s2.sum(); ratio = s2[:-1] / s2[1:]
    N = int(ratio.argmax()) + 1                                                 # N_sigma
    Kk = 10; k = np.arange(1, Kk + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    a1.bar(k, 100 * share[:Kk], color=['#C44E52' if i < N else '#8C8C8C' for i in range(Kk)])
    a1.set_xticks(k); a1.set_xlabel('Rank of singular value ($k$)'); a1.set_ylabel(r'Share of $\sigma_k^2$ in the total (%)')
    a1.set_title('(a) Squared singular values')
    a2.plot(k[:Kk - 1], ratio[:Kk - 1], 'o-', color='#4C72B0')
    a2.plot([N], [ratio[N - 1]], 'o', color='#C44E52', markersize=11, zorder=3)
    a2.annotate(rf'$N_\sigma={N}$', (N, ratio[N - 1]), textcoords='offset points', xytext=(14, -4), color='#C44E52')
    a2.set_xticks(k[:Kk - 1]); a2.set_xlabel('Rank of singular value ($k$)'); a2.set_ylabel(r'Ratio $\sigma_k^2/\sigma_{k+1}^2$')
    a2.set_title('(b) Ratio of consecutive squared singular values')
    for ax in (a1, a2):
        ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    save(fig, 'fig_sf_svd', exts=('png', 'pdf'))
    print(f'  N_sigma {N}, ratios {np.round(ratio[:6], 2)}, shares % {np.round(100 * share[:5], 1)}')


def step_mape(d):
    """test MAPE (%) per time step of the links with sensors, the links without sensors and the OD demand of one run"""
    lf, od = read_estimates(d)
    est, obs = lf.iloc[:, 0].values.reshape(T, -1, num_link), lf.iloc[:, 1].values.reshape(T, -1, num_link)
    ape = np.abs(est - obs) / (np.abs(obs) + 1e-8) * 100
    unobs = np.array(sorted(UNOBSERVED - 1)); obs_i = np.setdiff1d(np.arange(num_link), unobs)
    n = est.shape[1]; e_d, r_d = od.iloc[:, 0].values.reshape(T, n, -1), od.iloc[:, 1].values.reshape(T, n, -1)
    return ape[:, :, obs_i].mean((1, 2)), ape[:, :, unobs].mean((1, 2)), (np.abs(e_d - r_d) / (np.abs(r_d) + 1e-8)).mean((1, 2)) * 100


def fig_time_dynamics(S, d):
    """Fig. 16 (Sec. 7.2.3), the paper shows S = 5: (a)-(c) mean observed / estimated link flows by step (all links, with,
    without sensors); (d) mean reference / estimated OD demand; (e) link-flow MAPE per step (dotted: means);
    (f) OD-demand MAPE per step. Bands: +- 1 sd over the 100 test samples"""
    serif()
    NUM_LINK = num_link
    unobs_idx = np.array(sorted(UNOBSERVED - 1)); obs_idx = np.setdiff1d(np.arange(NUM_LINK), unobs_idx)
    FT, FL, FK, FG, FV = 21, 20, 17, 17, 15                             # font sizes: title, axis label, ticks, legend, bar values
    C_OBS, C_EST, C_REF, C_UNOBS = '#00A1DB', '#FA7F2C', '#7ACAB4', '#E85D5D'
    ts = np.arange(1, T + 1)

    def band(ax, y, c, lab, style):
        m, s = y.mean(axis=1), y.std(axis=1)
        ax.plot(ts, m, style, color=c, lw=2.2, ms=8, label=lab); ax.fill_between(ts, m - s, m + s, alpha=0.12, color=c)

    def finish(ax, title, ylabel, grid_axis='both'):
        ax.set_title(title, fontsize=FT); ax.set_xlabel('Time step', fontsize=FL); ax.set_ylabel(ylabel, fontsize=FL)
        ax.tick_params(labelsize=FK); ax.set_xticks(ts); ax.grid(True, alpha=0.3, axis=grid_axis)

    lf, od = read_estimates(d)
    est_flow = lf.iloc[:, 0].values.reshape(T, -1, NUM_LINK); obs_flow = lf.iloc[:, 1].values.reshape(T, -1, NUM_LINK)
    n = est_flow.shape[1]
    est_dem = od.iloc[:, 0].values.reshape(T, n, -1); ref_dem = od.iloc[:, 1].values.reshape(T, n, -1)

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.4))
    for ax, idx, title in ((axes[0, 0], np.arange(NUM_LINK), '(a) Link flow: all links'), (axes[0, 1], obs_idx, '(b) Link flow: with sensors'),
                           (axes[0, 2], unobs_idx, '(c) Link flow: without sensors')):
        band(ax, obs_flow[:, :, idx].mean(axis=2), C_OBS, 'Observed', 'o-'); band(ax, est_flow[:, :, idx].mean(axis=2), C_EST, 'Estimated', 's--')
        finish(ax, title, 'Mean link flow (veh/h)'); ax.legend(fontsize=FG, loc='upper left')

    ax = axes[1, 1]                                                   # (e) link-flow MAPE per time step
    ape = np.abs(est_flow - obs_flow) / (np.abs(obs_flow) + 1e-8) * 100
    m_o, m_u = ape[:, :, obs_idx].mean(axis=(1, 2)), ape[:, :, unobs_idx].mean(axis=(1, 2)); w = 0.38
    ax.bar(ts - w / 2, m_o, w, color=C_OBS, alpha=0.85, label='With sensors'); ax.bar(ts + w / 2, m_u, w, color=C_UNOBS, alpha=0.85, label='Without sensors')
    ax.axhline(ape[:, :, obs_idx].mean(), color=C_OBS, ls=':', alpha=0.7); ax.axhline(ape[:, :, unobs_idx].mean(), color=C_UNOBS, ls=':', alpha=0.7)
    ax.set_ylim(0, max(m_u.max(), m_o.max()) * 1.35)                    # no per-bar values (too crowded at this size); dotted = mean
    finish(ax, '(e) Link-flow MAPE per time step', 'MAPE (%)', 'y')
    ax.legend(fontsize=FG, ncol=2, loc='upper center')

    ax = axes[1, 0]                                                   # (d) OD demand
    band(ax, ref_dem.mean(axis=2), C_REF, 'Reference', 'o-'); band(ax, est_dem.mean(axis=2), C_EST, 'Estimated', 's--')
    finish(ax, '(d) OD demand', 'Mean OD demand (veh/h)'); ax.legend(fontsize=FG, loc='upper left')

    ax = axes[1, 2]                                                   # (f) OD-demand MAPE per time step
    od_t = (np.abs(est_dem - ref_dem) / (np.abs(ref_dem) + 1e-8)).mean(axis=(1, 2)) * 100
    ax.bar(ts, od_t, color=C_REF, alpha=0.85, edgecolor='#4a9e8a', lw=0.8)
    for i in range(T):
        ax.text(ts[i], od_t[i] + 0.15, f'{od_t[i]:.1f}', ha='center', va='bottom', fontsize=FV)
    ax.axhline(od_t.mean(), color=C_EST, ls='--', lw=1.6, label=f'Overall MAPE = {od_t.mean():.1f}%')
    ax.set_ylim(0, od_t.max() * 1.3); finish(ax, '(f) OD-demand MAPE per time step', 'MAPE (%)', 'y'); ax.legend(fontsize=FG, loc='upper center')

    fig.tight_layout(h_pad=1.5, w_pad=1.2)
    save(fig, f'fig_time_dynamics_s{S}_v2', dpi=250, quality=93)
    print(f'  test MAPE: sensor links {ape[:, :, obs_idx].mean():.2f}%, without sensors {ape[:, :, unobs_idx].mean():.2f}%, '
          f'all links {ape.mean():.2f}%, OD {od_t.mean():.2f}%')


def fig_link_error(d1, d5):
    """Fig. 17 (Sec. 7.2.4): (a)/(b) per-link test MAPE with S = 1 / S = 5 on the same colour scale (links without sensors
    dashed); (c) test link-flow MAPE by day type; (d) mean attention weight of each template by day type (S = 5)"""
    plt.rcdefaults(); plt.rc('font', family='Times New Roman'); plt.rcParams['mathtext.fontset'] = 'stix'
    node_coords = {1: (1, 10), 2: (5, 10), 3: (1, 8), 4: (2.5, 8), 5: (3.5, 8), 6: (5, 8), 7: (6.5, 7), 8: (5, 7), 9: (3.5, 7),
                   10: (3.5, 6), 11: (2.5, 6), 12: (1, 6), 13: (1, 2), 14: (2.5, 4), 15: (3.5, 4), 16: (5, 6), 17: (5, 5),
                   18: (6.5, 6), 19: (5, 4), 20: (5, 2), 21: (3.5, 2), 22: (3.5, 3), 23: (2.5, 3), 24: (2.5, 2)}
    G = nx.DiGraph(); edge_list = []
    for _, r in link_attributes.iterrows():
        G.add_edge(int(r['start']), int(r['end'])); edge_list.append((int(r['start']), int(r['end'])))
    is_unobs = np.isin(np.arange(1, num_link + 1), UNOBSERVED)
    typ = pd.read_csv(os.path.join(DATA, 'day_info.csv'))['type'].values[NTRAIN:]   # day type of the 100 test samples

    def load(d):
        x = pd.read_excel(os.path.join(d, 'link_flow_estimation.xlsx'))
        est = x.iloc[:, 0].values.reshape(T, -1, num_link); obs = x.iloc[:, 1].values.reshape(T, -1, num_link)
        return 100.0 * np.abs((est - obs) / obs)                                     # (T, 100, 76) absolute % errors

    def draw(ax, values, title, vmax):
        nx.draw_networkx_nodes(G, node_coords, ax=ax, node_size=800, node_color='white', edgecolors='black', linewidths=1.0)
        nx.draw_networkx_labels(G, node_coords, ax=ax, font_size=17, font_family='Times New Roman', font_weight='bold')
        for mask, style in ((~is_unobs, 'solid'), (is_unobs, 'dashed')):
            idx = np.where(mask)[0]
            nx.draw_networkx_edges(G, node_coords, edgelist=[edge_list[i] for i in idx], ax=ax,
                                   edge_color=[values[i] for i in idx], edge_cmap=plt.cm.Reds, edge_vmin=0, edge_vmax=vmax,
                                   width=2.5 if style == 'solid' else 3.0, style=style, arrows=True, arrowsize=10,
                                   connectionstyle='arc3,rad=0.1', alpha=0.9)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=vmax)); sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.02); cb.set_label('MAPE (%)', fontsize=24); cb.ax.tick_params(labelsize=20)
        ax.set_title(title, fontsize=27, pad=10); ax.axis('off')

    ape1, ape5 = load(d1), load(d5)
    link1, link5 = ape1.mean(axis=(0, 1)), ape5.mean(axis=(0, 1))                   # per-link MAPE
    vmax = float(np.ceil(link1.max()))
    att = np.load(os.path.join(d5, 'theta_attention.npz'))['attention_test']          # attention weights (T, 100, S)
    types = [1, 2, 3]
    by_type = {S: [a[:, typ == c].mean() for c in types] for S, a in ((1, ape1), (5, ape5))}
    att_type = np.array([att[:, typ == c].mean(axis=(0, 1)) for c in types])          # (3 day types, S templates)

    fig = plt.figure(figsize=(18, 19))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 0.75], left=0.03, right=0.97, top=0.96, bottom=0.05, wspace=0.28, hspace=0.22)
    ax = fig.add_subplot(gs[0, 0]); draw(ax, link1, '(a) Error with one template ($S=1$)', vmax)
    ax.legend(handles=[Line2D([0], [0], color='gray', lw=2.5, label='Link with sensor'),
                       Line2D([0], [0], color='gray', lw=3.0, ls='--', label='Link without sensor')],
              fontsize=20, loc='upper center', bbox_to_anchor=(0.45, 0.0), ncol=2, framealpha=0.95, edgecolor='gray')
    ax = fig.add_subplot(gs[0, 1]); draw(ax, link5, '(b) Error with five templates ($S=5$)', vmax)
    ax = fig.add_subplot(gs[1, 0])
    w = 0.36; xx = np.arange(3)
    for off, S, col in ((-w / 2, 1, '#8C8C8C'), (w / 2, 5, '#C44E52')):
        b = ax.bar(xx + off, by_type[S], w, color=col, label=f'$S={S}$')
        ax.bar_label(b, fmt='%.1f', fontsize=19, padding=2)
    ax.set_xticks(xx); ax.set_xticklabels([f'Day type {c}' for c in types], fontsize=21)
    ax.set_ylabel('Link-flow MAPE (%)', fontsize=24); ax.tick_params(labelsize=20)
    ax.set_ylim(0, max(by_type[1]) * 1.4); ax.legend(fontsize=21, frameon=False, ncol=2, loc='upper center'); ax.grid(axis='y', alpha=0.3)
    ax.set_title('(c) Link-flow error for each day type', fontsize=27, pad=10)
    ax = fig.add_subplot(gs[1, 1])                                    # heatmap: each row (day type) sums to 1
    im = ax.imshow(att_type, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    for i in range(att_type.shape[0]):
        for j in range(att_type.shape[1]):
            ax.text(j, i, f'{att_type[i, j]:.2f}', ha='center', va='center', fontsize=21, color='white' if att_type[i, j] > 0.5 else 'black')
    ax.set_xticks(range(att_type.shape[1])); ax.set_xticklabels([f'T{s}' for s in range(1, att_type.shape[1] + 1)], fontsize=21); ax.set_xlabel('Template', fontsize=24)
    ax.set_yticks(range(3)); ax.set_yticklabels([f'Day type {c}' for c in types], fontsize=21)
    cb = plt.colorbar(im, ax=ax, shrink=0.9, pad=0.02); cb.set_label('Mean attention weight', fontsize=24); cb.ax.tick_params(labelsize=20)
    ax.set_title('(d) Attention weights ($S=5$)', fontsize=27, pad=10)
    save(fig, 'fig_sf_link_error_D')
    print(f'  MAPE by day type: S=1 {np.round(by_type[1], 2)}, S=5 {np.round(by_type[5], 2)}; attention by day type (rows) '
          f'and template (columns):\n{att_type.round(2)}')


def fig_scatter(S, d):
    """Fig. 18 (Sec. 7.2.4), panel (a) S = 1, (b) S = 5: estimated vs observed link flows (left) and estimated vs
    reference OD trips (right), test samples; R^2 in the corner"""
    serif(16)
    COL_LINK, COL_OD, COL_LINE = '#4C72B0', '#DD8452', '#333333'

    def panel(ax, obs, est, col, xlabel, ylabel, title):
        lo, hi = min(obs.min(), est.min()), max(obs.max(), est.max())
        pad = 0.03 * (hi - lo); lo, hi = lo - pad, hi + pad
        ax.scatter(obs, est, s=2, color=col, alpha=0.15, linewidths=0, rasterized=True)
        ax.plot([lo, hi], [lo, hi], ls='--', color=COL_LINE, lw=1.4, label='$y=x$')
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal')
        ax.grid(True, alpha=0.35, lw=0.8)
        ax.set_xlabel(xlabel, fontsize=19); ax.set_ylabel(ylabel, fontsize=19); ax.tick_params(labelsize=16)
        r2 = 1 - ((est - obs) ** 2).sum() / ((obs - obs.mean()) ** 2).sum()
        ax.text(0.04, 0.95, f'$R^2={r2:.3f}$', transform=ax.transAxes, va='top', fontsize=17,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.7', alpha=0.9))
        ax.legend(loc='lower right', fontsize=16, framealpha=0.9)
        ax.set_title(title, fontsize=20, pad=8)

    lf, od = read_estimates(d)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6.4))
    panel(a1, lf.iloc[:, 1].values, lf.iloc[:, 0].values, COL_LINK, 'Observed link flow (veh/h)', 'Estimated link flow (veh/h)', 'Link flow: estimated vs. observed')
    panel(a2, od.iloc[:, 1].values, od.iloc[:, 0].values, COL_OD, 'Reference OD trips (veh/h)', 'Estimated OD trips (veh/h)', 'OD trips: estimated vs. reference')
    fig.tight_layout()
    save(fig, f'fig_scatter_s{S}_new')


def fig_convergence(d):
    """Fig. 19 (Sec. 7.2.4): training and test losses of the OD-weighted S = 5 run: (a) L_V, (b) L_Q, (c) L_D, (d) total
    loss; raw values thin and light, 25-iteration running mean bold; log scale except (c)"""
    serif(16)
    def rmean(x, w=25): return pd.Series(x).rolling(w, center=True, min_periods=1).mean().values
    tr = pd.read_excel(os.path.join(d, 'loss_train.xlsx')); te = pd.read_excel(os.path.join(d, 'loss_test.xlsx'))
    it = np.arange(1, len(te) + 1)
    panels = [('link_flow', r'(a) Link-flow loss $L_V$'), ('demand', r'(b) Demand loss $L_Q$'),
              ('vdf', r'(c) VDF loss $L_D$'), ('total', r'(d) Total loss')]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    for ax, (col, title) in zip(axes.ravel(), panels):
        for df, c, lab in ((tr, '#C44E52', 'Training'), (te, '#4C72B0', 'Test')):
            y = df[col].values
            ax.plot(it, y, color=c, lw=0.6, alpha=0.25, label=f'{lab}: loss at each iteration')
            ax.plot(it, rmean(y), color=c, lw=2.2, label=f'{lab}: smoothed (25-iteration running mean)')
        if col == 'vdf':                                            # nearly constant: linear scale, range around the data
            lo, hi = np.percentile(np.r_[tr[col].values, te[col].values], [0.5, 99.5]); pad = 0.3 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)
        else:
            ax.set_yscale('log')
        ax.set_xlim(0, len(te))
        ax.set_title(title, fontsize=19); ax.set_xlabel('Iteration', fontsize=18); ax.set_ylabel('Loss', fontsize=18)
        ax.grid(True, which='major', alpha=0.35)
    h, l = axes[0, 0].get_legend_handles_labels()
    for x in h:
        x.set_alpha(min(1, 3 * x.get_alpha() if x.get_alpha() else 1))
    fig.legend(h, l, loc='upper center', ncol=2, fontsize=15, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    save(fig, 'fig_sf_convergence_od_weighted')


if __name__ == '__main__':
    table5()
    table6()
    table7()
    fig_od_nodes()
    fig_svd()
    d1, d5, d5w = best_run(1), best_run(5), best_run(5, odw=True)
    print('runs used for the figures (lowest training sensor MAPE):', {'S=1': d1, 'S=5': d5, 'S=5 OD-weighted': d5w})
    have = lambda d: d is not None and os.path.exists(os.path.join(d, 'link_flow_estimation.xlsx'))
    if have(d5):
        fig_time_dynamics(5, d5)
    else:
        print('skipped fig_time_dynamics_s5_v2 (needs the test estimates of a run with S = 5)')
    for S, d in ((1, d1), (5, d5)):                 # per-step MAPE quoted in the text of Sec. 7.2.3 (S = 1 vs S = 5)
        if have(d):
            m_o, m_u, m_q = step_mape(d)
            print(f'  S={S} MAPE per step (%): with sensors {np.round(m_o, 1)}, without sensors {np.round(m_u, 1)}, OD {np.round(m_q, 1)}')
    if have(d1) and have(d5):
        fig_link_error(d1, d5)
    else:
        print('skipped fig_sf_link_error_D (needs the test estimates of a run with S = 1 and one with S = 5)')
    for S, d in ((1, d1), (5, d5)):
        if have(d):
            fig_scatter(S, d)
        else:
            print(f'skipped fig_scatter_s{S}_new (needs the test estimates of a run with S = {S})')
    if d5w and os.path.exists(os.path.join(d5w, 'loss_test.xlsx')):
        fig_convergence(d5w)
    else:
        print('skipped fig_sf_convergence_od_weighted (needs the losses of an OD-weighted run with S = 5)')
