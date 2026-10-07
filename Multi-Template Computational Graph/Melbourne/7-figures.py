"""Step 7: figures of the Melbourne case (paper Section 7.3, Figs. 21-26), png + pdf.

Purpose
-------
Draws the six Melbourne figures of the paper from the package's own results (steps 4-6). The drawing code is the
code of the paper's figure scripts (gen_fig_melb_templates.py, gen_fig_melb_results.py, gen_fig_melb_error_time.py,
gen_fig_melb_map.py); only the file locations differ (all relative to this package), and the search steps of
Fig. 21(b) are read from the output of step 4 instead of being typed in.

  fig_melb_templates    Fig. 21 (Sec. 7.3.3) (a) shares of the squared singular values sigma_k^2 of the centred
                        training counts and N_sigma = argmax_k sigma_k^2 / sigma_{k+1}^2 (Sec. 5.1, Eq. 46);
                        (b) theta-distance D (Sec. 5.2, Eq. 49) during the swap search for three of the ten
                        candidates, from {T5, T7, T8} and {T4, T6, T9}; dashed: best of all 120 sets;
                        (c) theta-distance by time step with Template 1 alone and with Templates 1-3, by day type
  fig_melb_convergence  Fig. 22 (Sec. 7.3.4) block-wise training of S = 3 (Section 6): losses L_V, L_Q, L_D and the
                        fused VOT (Eq. 53) against the benchmark VOT; shading = block being updated
  fig_melb_error_time   Fig. 23 (Sec. 7.3.5) link-flow RMSE of S = 1, 2, 3 per time step: all road links (4,223, vs
                        noise-free flows) and links without sensors (45 withheld sensors, vs counts); all test days and
                        stadium-event days  (+ fig_melb_error_time.csv)
  fig_melb_vot          Fig. 24 (Sec. 7.3.6) (a) fused VOT per time step vs the benchmark VOT, S = 1, 2, 3, with the
                        MAPE over the 12 steps; (b) bias of the fused VOT by day type  (+ fig_melb_vot_accuracy.csv)
  fig_melb_attention    Fig. 25 (Sec. 7.3.6) attention weights lambda_{s,t} of S = 3: (a) mean by day type;
                        (b) stadium-event days by time step; one colour per template
  fig_melb_map          Fig. 26 (Sec. 7.3.6) inner city, 9:00-9:45, 2 x 2: rows stadium-event / regular test days;
                        columns error reduction of S = 3 over S = 1 / flow change by Templates 2-3 (fused flow minus
                        the flow of Template 1 alone)  (+ fig_melb_map_bands.csv: error by distance to the stadiums)
Also printed: the R^2 of the test link flows of S = 1 and S = 3 (all links, entries above one vehicle; Sec. 7.3.5) and
the numbers of the map paragraph. Error conventions: MAPE over entries > 1; RMSE in veh/15 min. A figure whose inputs
are missing is skipped with a message.

Usage
-----
    python 7-figures.py                         (about 1 min)
    python 7-figures.py --results <folder with the MTCG run folders> --tables <folder with M*.csv> --out <folder>
    e.g. the paper's runs:  python 7-figures.py --results ../Melbourne_r2_v2/results --tables reference_results/tables

Inputs
------
    data/days100_logit_start7_cap4_k2/ (step 3); results/template_selection/ (step 4: theta_distance_by_slot.csv,
    selection_search.csv, selection_exhaustive.csv); results/mtcg_S{1,2,3}/ (step 5: predictions.npz,
    training_history.csv); results/tables/M3b, M3c, M4c, M4d (step 6); data/network/link_xy.csv, node_xy.csv
Outputs
-------
    results/figures/fig_melb_*.png / .pdf and the csv files named above

Authors:
    Xin (Bruce) Wu, Department of Civil and Environmental Engineering, Villanova University, PA, USA
    Feng Shao, School of Mathematics, China University of Mining and Technology, China

Contact: xwu03@villanova.edu (Villanova University), xinwu8592@gmail.com (personal)

MIT License
Copyright (c) 2026 Xin (Bruce) Wu, Feng Shao
"""
import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import mtcg_common as C

ap = argparse.ArgumentParser()
ap.add_argument('--results', default=C.RES, help='folder that holds the MTCG run folders')
ap.add_argument('--tables', default=os.path.join(C.RES, 'tables'))
ap.add_argument('--selection', default=os.path.join(C.RES, 'template_selection'))
ap.add_argument('--out', default=os.path.join(C.RES, 'figures'))
args = ap.parse_args()
OUTF = args.out; os.makedirs(OUTF, exist_ok=True)
RES = {S: C.mtcg_run_dir(S, args.results) for S in (1, 2, 3)}
TAB = args.tables
DATA = C.DAYS
T = 12; CLOCK = C.CLOCK
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'mathtext.fontset': 'stix'})
FT, FL, FK, FG = 21, 20, 16, 16                                                # font sizes: title, label, ticks, legend


def save(fig, name, dpi=250):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUTF, f'{name}.{ext}'), dpi=dpi, bbox_inches='tight')
    plt.close(fig); print('saved', name)


def have(name, *S):
    miss = [s for s in S if RES[s] is None]
    if miss:
        print(f'skipped {name}: no MTCG results for S = {miss} (run 5-mtcg.py)')
    return not miss


def need_files(name, *files):
    miss = [f for f in files if not os.path.exists(f)]
    if miss:
        print(f'skipped {name}: missing {", ".join(os.path.relpath(f, C.ROOT) for f in miss)}')
    return not miss


def ape(est, ref):
    """Absolute percentage error, NaN where the reference is at most one vehicle."""
    m = np.abs(ref) > 1
    return np.where(m, np.abs(est - ref) / np.where(m, np.abs(ref), 1), np.nan) * 100


te = np.load(os.path.join(DATA, 'test', 'days.npz'))
true_all = te['link_flow_all'][:, 4:16, :].transpose(1, 0, 2)                # (12 steps, 20 days, 7615 links) noise-free
typ = pd.read_csv(os.path.join(DATA, 'test', 'days.csv')).type.values         # day type of each test day
xy = pd.read_csv(os.path.join(C.NETWORK, 'link_xy.csv'))
ROAD = xy.is_connector.values == 0                                            # 4,223 road links (connectors excluded)
Z = {S: np.load(os.path.join(RES[S], 'predictions.npz')) for S in (1, 2, 3) if RES[S] is not None}

# Candidate names of steps 2/4 -> numbering of the paper (Table 10)
PAPER_NAME = {'T1': 'T1', 'T2': 'T2', 'T3': 'T3', 'lvl080': 'T4', 'lvl120': 'T5', 'stad2x': 'T6', 'stad_early': 'T7',
              'poi_late': 'T8', 'other_event': 'T9', 'both': 'T10'}


# =================================================================================================================
# Fig. 21 fig_melb_templates (Section 7.3.3)
# =================================================================================================================
def swap_paths(theta_label='0.5 x wage'):
    """Swap searches of step 4 from {T5, T7, T8} and {T4, T6, T9}: [(label, colour, marker, [(D %, 'out→in'), ...])],
    the best and the worst D of all 120 sets. D rounded to two decimals, as printed in the paper."""
    sr = pd.read_csv(os.path.join(args.selection, 'selection_search.csv'))
    ex = pd.read_csv(os.path.join(args.selection, 'selection_exhaustive.csv'))
    sr, ex = sr[(sr.theta_label == theta_label) & (sr.method == 'swaps')], ex[ex.theta_label == theta_label]
    dist_of = {frozenset(s.split('+')): v for s, v in zip(ex.templates, ex['mean_distance_%'])}
    out = []
    for start, c, mk in (('lvl120+stad_early+poi_late', '#4C72B0', 's'), ('lvl080+other_event+stad2x', '#55A868', '^')):
        r = sr[sr.start == start].iloc[0]
        names = start.split('+')
        steps = [(round(dist_of[frozenset(names)], 2), '')]
        for a, b, v in re.findall(r'(\S+) -> (\S+) \(([\d.]+)%\)', r.steps):
            steps.append((float(v), f'{PAPER_NAME[a]}→{PAPER_NAME[b]}'))
        lab = 'Start {' + ', '.join(sorted((PAPER_NAME[n] for n in names), key=lambda x: int(x[1:]))) + '}'
        out.append((lab, c, mk, steps))
    v = ex['mean_distance_%'].values
    return out, round(float(v.min()), 2), round(float(v.max()), 2)


def fig_templates():
    sel = args.selection
    if not need_files('fig_melb_templates', *(os.path.join(sel, f) for f in
                      ('theta_distance_by_slot.csv', 'selection_search.csv', 'selection_exhaustive.csv'))):
        return
    with plt.rc_context({'font.size': 18}):
        z = np.load(os.path.join(DATA, 'train', 'days.npz'))
        x = z['link_flow'][:, 4:16, :].astype(float)                               # (80 days, 12 slots, 448 sensors)
        x = x - x.mean(0, keepdims=True)                                           # remove the mean of each slot (Eq. 45)
        s2 = np.linalg.svd(x.reshape(-1, x.shape[2]), compute_uv=False) ** 2       # sigma_k^2 of the 960 x 448 matrix
        s2 = s2[s2 > 1e-9 * s2[0]]                                                 # drop zero singular values
        share = s2 / s2.sum(); ratio = s2[:-1] / s2[1:]
        N = int(ratio.argmax()) + 1                                                # N_sigma (Eq. 46)
        K = 8; k = np.arange(1, K + 1)
        td = pd.read_csv(os.path.join(sel, 'theta_distance_by_slot.csv'))
        clock = list(td.columns[2:]); ts = np.arange(len(clock))
        SEARCH, BEST, WORST = swap_paths()

        fig = plt.figure(figsize=(14, 10.5)); gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])
        a1, a2, a3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
        a1.bar(k, 100 * share[:K], color=['#C44E52' if i < N else '#8C8C8C' for i in range(K)])
        a1.set_xticks(k); a1.set_xlabel('Rank of singular value ($k$)'); a1.set_ylabel(r'Share of $\sigma_k^2$ in the total (%)')
        a1.text(0.97, 0.95, rf'$N_\sigma={N}$' + '\n' + rf'$\sigma_{N}^2/\sigma_{N + 1}^2={ratio[N - 1]:.1f}$', transform=a1.transAxes,
                ha='right', va='top', color='#C44E52', fontsize=18)
        a1.set_title('(a) Squared singular values')
        for j, (lab, c, mk, steps) in enumerate(SEARCH):
            d = [v for v, _ in steps]; it = np.arange(len(d))
            a2.plot(it, d, '-', marker=mk, color=c, lw=2, ms=9, label=lab)
            a2.plot([3, 4], [d[-1], d[-1]], ':', color=c, lw=2)                   # step 4: no swap reduces D -> stop
            a2.plot([4], [d[-1]], marker=mk, color=c, ms=10, mfc='white', mew=2)
            for i in range(1, 4):                                                 # template swapped in at each step
                a2.annotate(steps[i][1], (it[i], d[i]), textcoords='offset points', xytext=(4, 10 + 16 * j), fontsize=12, color=c)
        a2.annotate('stop: no swap\nreduces $D$', (4, BEST), textcoords='offset points', xytext=(-70, 70), fontsize=14,
                    arrowprops=dict(arrowstyle='->', color='k', lw=1.2))
        a2.axhline(BEST, color='k', ls='--', lw=1.3, label=f'Best of all 120 sets ({BEST:.2f}%)')
        a2.set_xticks(range(5)); a2.set_xlabel('Iteration of the swap search'); a2.set_ylabel(r'$\theta$-distance $D$ (%)')
        a2.set_ylim(5.0, 8.1); a2.legend(fontsize=13, loc='upper right', framealpha=0.9)
        a2.set_title(r'(b) Search for three of ten candidates')
        for dt, c, lab in (('stadium', '#C44E52', 'Stadium days'), ('poi', '#4C72B0', 'POI days'), ('regular', '#8C8C8C', 'Regular days')):
            for tpl, ls, mk in (('T1', '--', 's'), ('T1+T2+T3', '-', 'o')):
                r = td[(td.templates == tpl) & (td.day_type == dt)].iloc[0, 2:].values.astype(float)
                a3.plot(ts, r, ls, marker=mk, color=c, lw=1.8, ms=6, label=f'{lab}, {"Template 1 only" if tpl == "T1" else "Templates 1-3"}')
        a3.set_xticks(ts); a3.set_xticklabels(clock); a3.set_xlabel('Time step (start time)')
        a3.set_ylabel(r'$\theta$-distance (%)'); a3.set_title(r'(c) $\theta$-distance of the observations by time step')
        a3.legend(fontsize=14, ncol=3, loc='upper left', framealpha=0.9)
        a3.set_ylim(4.5, 11.5)
        for ax in (a1, a2, a3):
            ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        save(fig, 'fig_melb_templates', dpi=300)
        print('  N_sigma', N, 'shares %', np.round(100 * share[:4], 1), 'ratio', round(float(ratio[N - 1]), 2), 'best', BEST, 'worst set', WORST)


# =================================================================================================================
# Fig. 22 fig_melb_convergence (Section 7.3.4)
# =================================================================================================================
def fig_convergence():
    if not have('fig_melb_convergence', 3):
        return
    h = pd.read_csv(os.path.join(RES[3], 'training_history.csv'))
    ep = h.epoch.values + 1
    # blocks of Section 6: D = Step 1 demand, TH = Step 2 route choice (theta), A = Step 3 attention, J = Step 4 fine-tuning
    BC = {'D': ('#F2C14E', 'Demand block'), 'TH': ('#7ACAB4', r'$\theta$ block'), 'A': ('#9DB4E0', 'Attention block'), 'J': ('#D9A6C6', 'All blocks')}
    vot_bench = 24.3                                                              # benchmark VOT, whole period (Table 13)
    f_m3 = os.path.join(TAB, 'M3_vot_by_period.csv')
    if os.path.exists(f_m3):
        vot_bench = round(float(pd.read_csv(f_m3, encoding='utf-8-sig')['Benchmark VOT'].iloc[0]), 1)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, (cols, title, ylab, logy) in zip(axes.ravel(), (
            (('train_link', 'test_link'), '(a) Link-flow loss $L_V$', 'Loss', True),
            (('train_demand', 'test_demand'), '(b) Demand loss $L_Q$', 'Loss', True),
            (('train_vdf', 'test_vdf'), '(c) VDF loss $L_D$', 'Loss', False),
            (('fused_vot',), '(d) Fused VOT (training days)', 'VOT (AUD/h)', False))):
        for b0, b1, blk in [(g.epoch.min() + 0.5, g.epoch.max() + 1.5, k) for (rd, k), g in h.groupby(['round', 'block'], sort=False)]:
            ax.axvspan(b0, b1, color=BC[blk][0], alpha=0.25, lw=0)
        for col, c, lab in zip(cols, ('#C44E52', '#4C72B0'), ('Training', 'Test')):
            ax.plot(ep, h[col].values, color=c, lw=2, label=lab if col != 'fused_vot' else 'Fused VOT')
        if cols[0] == 'fused_vot':
            ax.axhline(vot_bench, color='k', ls='--', lw=1.4, label=f'Benchmark VOT {vot_bench}')
        if logy:
            ax.set_yscale('log')
        ax.set_title(title, fontsize=19); ax.set_xlabel('Epoch', fontsize=17); ax.set_ylabel(ylab, fontsize=17); ax.tick_params(labelsize=14)
        ax.grid(alpha=0.3); ax.set_xlim(0.5, ep.max() + 0.5); ax.legend(fontsize=13, loc='upper right')
    fig.legend(handles=[Patch(color=v[0], alpha=0.5, label=v[1]) for v in BC.values()], loc='upper center', ncol=4, fontsize=15, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(); save(fig, 'fig_melb_convergence')


# =================================================================================================================
# Fig. 23 fig_melb_error_time (Section 7.3.5)
# =================================================================================================================
def fig_error_time():
    if not have('fig_melb_error_time', 1, 2, 3):
        return
    ts = np.arange(T)
    EV = np.isin(typ, ['stadium', 'both'])                                        # stadium-event test days
    CS = {1: '#8C8C8C', 2: '#4C72B0', 3: '#C44E52'}; MK = {1: 's', 2: '^', 3: 'o'}

    def rmse_t(e, o):
        return np.sqrt(((e - o) ** 2).mean(axis=(1, 2)))                          # (12,)
    curves = {}
    for S in (1, 2, 3):
        z = Z[S]
        p = z['test_link_pred']; c = z['test_link_obs']; ps = p[:, :, z['observation_link_number']]; u = z['unobserved_link_idx']
        for days, key in ((np.ones(len(typ), bool), 'all'), (EV, 'ev')):
            curves[('road', key, S)] = rmse_t(p[:, days][:, :, ROAD], true_all[:, days][:, :, ROAD])   # all road links
            curves[('wo', key, S)] = rmse_t(ps[:, days][:, :, u], c[:, days][:, :, u])                # links without sensors
    with plt.rc_context({'font.size': 17}):
        fig, axes = plt.subplots(2, 2, figsize=(16, 10.5), sharex=True)
        PANELS = [(('road', 'all'), '(a) All road links, all test days'), (('road', 'ev'), '(b) All road links, stadium-event days'),
                  (('wo', 'all'), '(c) Links without sensors, all test days'), (('wo', 'ev'), '(d) Links without sensors, stadium-event days')]
        for ax, ((grp, key), title) in zip(axes.ravel(), PANELS):
            ax.axvspan(-0.4, 2.4, color='#7ACAB4', alpha=0.15, lw=0); ax.axvspan(6.6, 10.4, color='gold', alpha=0.18, lw=0)
            for S in (1, 2, 3):
                ax.plot(ts, curves[(grp, key, S)], '-', marker=MK[S], color=CS[S], lw=2, ms=7, label=f'$S={S}$')
            ax.set_title(title, fontsize=19); ax.set_ylabel('RMSE (veh/15 min)'); ax.grid(alpha=0.3); ax.set_ylim(0, None)
            ax.set_xticks(ts); ax.set_xticklabels(CLOCK, rotation=45)
        axes[0, 0].legend(loc='lower left', fontsize=15)
        ymax = axes[0, 0].get_ylim()[1]
        axes[0, 0].text(1, ymax * 0.93, 'POI trips', ha='center', fontsize=14); axes[0, 0].text(8.5, ymax * 0.93, 'stadium trips', ha='center', fontsize=14)
        for ax in axes[1]:
            ax.set_xlabel('Time step (start time)')
        fig.tight_layout()
        save(fig, 'fig_melb_error_time')
    out = pd.DataFrame({f'{g}-{k}-S{S}': v for (g, k, S), v in curves.items()}, index=CLOCK)
    out.round(2).to_csv(os.path.join(OUTF, 'fig_melb_error_time.csv'))
    print(out.round(1).to_string())


# =================================================================================================================
# Fig. 24 fig_melb_vot and Fig. 25 fig_melb_attention (Section 7.3.6)
# =================================================================================================================
def fig_vot():
    f_t, f_d = os.path.join(TAB, 'M3b_vot_by_timestep.csv'), os.path.join(TAB, 'M3c_vot_by_daytype.csv')
    if not need_files('fig_melb_vot (run 6-tables.py)', f_t, f_d):
        return
    vt = pd.read_csv(f_t, encoding='utf-8-sig'); vd = pd.read_csv(f_d, encoding='utf-8-sig')
    if not all(f'S={S}' in vt.columns for S in (1, 2, 3)):
        print('skipped fig_melb_vot: the tables need S = 1, 2 and 3'); return
    ts = np.arange(1, T + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 5.6), gridspec_kw={'width_ratios': [1.5, 1]})
    a1.plot(ts, vt['Benchmark VOT'], 'k-', lw=2.6, label='Benchmark VOT')
    a1.plot(ts, vt['Benchmark VOT, stadium event days'], 'k:', lw=1.8, label='Benchmark VOT, stadium-event days')
    # accuracy over the 12 time steps: MAPE and correlation of the mean fused VOT against the benchmark VOT
    CV = {1: '#8C8C8C', 2: '#4C72B0', 3: '#D62728'}                            # S = 3 in red
    acc = []
    for S in (1, 2, 3):
        e, b = vt[f'S={S}'].values, vt['Benchmark VOT'].values
        acc.append({'S': S, 'MAPE %': np.mean(np.abs(e - b) / b) * 100, 'corr': np.corrcoef(e, b)[0, 1]})
    acc = pd.DataFrame(acc).set_index('S'); acc.round(3).to_csv(os.path.join(OUTF, 'fig_melb_vot_accuracy.csv'))
    for S, mk in ((1, 's'), (2, '^'), (3, 'o')):
        st = dict(ls='-', lw=3.2, ms=8, zorder=5) if S == 3 else dict(ls='--', lw=1.6, ms=6, zorder=3)
        a1.plot(ts, vt[f'S={S}'], marker=mk, color=CV[S], label=f'Fused VOT, $S={S}$ (MAPE {acc.loc[S, "MAPE %"]:.1f}%)', **st)
    a1.axvspan(8.5, 11.5, color='gold', alpha=0.15, lw=0); a1.text(10, 36.5, 'stadium trips', ha='center', fontsize=13)
    a1.axvspan(0.5, 3.5, color='#7ACAB4', alpha=0.15, lw=0); a1.text(2, 36.5, 'POI trips', ha='center', fontsize=13)
    a1.set_xticks(ts); a1.set_xticklabels(CLOCK, rotation=45); a1.set_ylim(14, 38)
    a1.set_xlabel('Time step (start time)', fontsize=FL - 2); a1.set_ylabel('VOT (AUD/h)', fontsize=FL - 2); a1.tick_params(labelsize=FK - 2)
    a1.set_title('(a) Fused VOT per time step (training days)', fontsize=FT - 2); a1.grid(alpha=0.3); a1.legend(fontsize=12, ncol=2, loc='lower center')
    x = np.arange(len(vd)); w = 0.26
    for k, S in enumerate((1, 2, 3)):
        a2.bar(x + (k - 1) * w, vd[f'S={S} bias %'], w, color=CV[S], label=f'$S={S}$')
    a2.axhline(0, color='k', lw=1); a2.set_xticks(x); a2.set_xticklabels(['Regular', 'POI', 'Stadium', 'Stadium\n+ POI'])
    a2.set_ylabel('Bias of the fused VOT (%)', fontsize=FL - 2); a2.tick_params(labelsize=FK - 2)
    a2.set_title('(b) Bias by day type', fontsize=FT - 2); a2.grid(axis='y', alpha=0.3); a2.legend(fontsize=13)
    fig.tight_layout(); save(fig, 'fig_melb_vot')


def fig_attention():
    f_c, f_d = os.path.join(TAB, 'M4c_attention_by_daytype.csv'), os.path.join(TAB, 'M4d_attention_stadium_days.csv')
    if not need_files('fig_melb_attention (run 6-tables.py)', f_c, f_d):
        return
    ac = pd.read_csv(f_c, encoding='utf-8-sig'); ad = pd.read_csv(f_d, encoding='utf-8-sig')
    if 'run' not in ac.columns or not (ac.run == 'S=3').any():
        print('skipped fig_melb_attention: no S = 3 rows'); return
    ac = ac[ac.run == 'S=3']; ad = ad[ad.run == 'S=3']
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(17, 5.4), gridspec_kw={'width_ratios': [1, 2.4]})
    # one colour per template, each scaled from half its minimum to its maximum, so the small weights of T2 and T3 stay visible
    TCM = {0: plt.cm.Reds, 1: LinearSegmentedColormap.from_list('yel', ['#FFFDF0', '#FFE680', '#F2C200', '#C99A00']), 2: plt.cm.Greens}

    def cells(ax, W, by_row):
        # W: matrix of weights lambda; template index along rows (by_row) or along columns
        rgba = np.zeros(W.shape + (4,))
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                k = i if by_row else j
                w = W[k, :] if by_row else W[:, k]
                lo = 0.5 * w.min(); z = (W[i, j] - lo) / (w.max() - lo)
                rgba[i, j] = TCM[k](0.08 + 0.85 * z)
                ax.text(j, i, f'{W[i, j]:.2f}', ha='center', va='center', fontsize=20,
                        color='white' if (z > 0.6 and k != 1) else 'k')
        ax.imshow(rgba, aspect='auto')

    M = ac[['T1', 'T2', 'T3']].values
    cells(a1, M, by_row=False)
    a1.set_xticks(range(3)); a1.set_xticklabels(['T1', 'T2', 'T3']); a1.set_yticks(range(len(ac))); a1.set_yticklabels(['Regular', 'POI', 'Stadium', 'Stadium + POI'])
    a1.set_xlabel('Template', fontsize=20); a1.set_title('(a) Mean weight by day type', fontsize=22); a1.tick_params(labelsize=18)
    M2_ = ad[CLOCK].values
    cells(a2, M2_, by_row=True)
    a2.set_xticks(range(T)); a2.set_xticklabels(CLOCK); a2.set_yticks(range(3)); a2.set_yticklabels(['T1', 'T2', 'T3'])
    a2.set_xlabel('Time step (start time)', fontsize=20); a2.set_title('(b) Stadium-event days, by time step', fontsize=22); a2.tick_params(labelsize=18)
    fig.tight_layout()
    save(fig, 'fig_melb_attention')


# =================================================================================================================
# Fig. 26 fig_melb_map (Section 7.3.6)
# =================================================================================================================
def fig_map():
    if not have('fig_melb_map', 1, 3):
        return
    nd = pd.read_csv(os.path.join(C.NETWORK, 'node_xy.csv')).set_index('node_id')
    p1 = Z[1]['test_link_pred']
    p3 = Z[3]['test_link_pred']; t1 = Z[3]['test_link_pred_per_template'][0]  # fused flows and Template 1 alone (S = 3)
    STEPS = [8, 9, 10, 11]                                                     # 9:00, 9:15, 9:30, 9:45
    ev, reg = np.isin(typ, ['stadium', 'both']), typ == 'regular'

    def mean_over(a, days):
        return a[np.ix_(STEPS, days)].mean(axis=(0, 1))
    gain = mean_over(np.abs(p1 - true_all), ev) - mean_over(np.abs(p3 - true_all), ev)          # error reduction
    gain_reg = mean_over(np.abs(p1 - true_all), reg) - mean_over(np.abs(p3 - true_all), reg)
    contrib_ev, contrib_reg = mean_over(p3 - t1, ev), mean_over(p3 - t1, reg)                    # flow change by T2-3

    keep = ROAD
    seg = np.stack([xy[['x_from', 'y_from']].values, xy[['x_to', 'y_to']].values], axis=1) / 1000   # km
    X0, X1, Y0, Y1 = 316.5, 327.5, 5806.5, 5817.5                              # inner-city window (UTM, km)
    inwin = keep & (seg[:, :, 0].min(1) > X0) & (seg[:, :, 0].max(1) < X1) & (seg[:, :, 1].min(1) > Y0) & (seg[:, :, 1].max(1) < Y1)
    stad, poi = nd.loc[2341:2346][['x_coord', 'y_coord']].values / 1000, nd.loc[2347:2348][['x_coord', 'y_coord']].values / 1000

    def draw(ax, val, title, lim, label):
        norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
        ax.add_collection(LineCollection(seg[inwin], colors='#DDDDDD', linewidths=0.5, zorder=1))
        order = np.argsort(np.abs(val[inwin]))                                 # strongest values on top
        v = val[inwin][order]
        lc = LineCollection(seg[inwin][order], cmap='RdBu_r', norm=norm, linewidths=0.6 + 2.4 * np.clip(np.abs(v) / lim, 0, 1), zorder=2)
        lc.set_array(v); ax.add_collection(lc)
        ax.scatter(stad[:, 0], stad[:, 1], marker='*', s=260, c='gold', edgecolors='k', linewidths=0.8, zorder=4, label='Stadium')
        ax.scatter(poi[:, 0], poi[:, 1], marker='D', s=80, c='white', edgecolors='k', linewidths=0.8, zorder=4, label='POI')
        ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=24)
        cb = plt.colorbar(lc, ax=ax, shrink=0.78, pad=0.02); cb.set_label(label, fontsize=19); cb.ax.tick_params(labelsize=17)
        ax.plot([X0 + 0.5, X0 + 2.5], [Y0 + 0.6, Y0 + 0.6], 'k-', lw=3); ax.text(X0 + 1.5, Y0 + 0.9, '2 km', ha='center', fontsize=18)

    with plt.rc_context({'font.size': 20}):
        lim_g = np.percentile(np.abs(np.r_[gain[inwin], gain_reg[inwin]]), 98); lim_c = np.percentile(np.abs(np.r_[contrib_ev[inwin], contrib_reg[inwin]]), 98)
        fig, axes = plt.subplots(2, 2, figsize=(16, 15))
        draw(axes[0, 0], gain, '(a) Error reduction, stadium-event days', lim_g, 'Error reduction of $S=3$ (veh/15 min)')
        draw(axes[0, 1], contrib_ev, '(b) Flow change, stadium-event days', lim_c, 'Flow change by Templates 2-3 (veh/15 min)')
        draw(axes[1, 0], gain_reg, '(c) Error reduction, regular days', lim_g, 'Error reduction of $S=3$ (veh/15 min)')
        draw(axes[1, 1], contrib_reg, '(d) Flow change, regular days', lim_c, 'Flow change by Templates 2-3 (veh/15 min)')
        axes[0, 0].legend(loc='upper left', fontsize=18, framealpha=0.9)
        fig.tight_layout()
        save(fig, 'fig_melb_map')

    # numbers for the text: mean absolute error by distance to the nearest stadium (event days, 9:00-9:45, no connectors)
    mid = seg.mean(1); dist = np.min(np.hypot(mid[:, None, 0] - stad[None, :, 0], mid[:, None, 1] - stad[None, :, 1]), axis=1)
    rows = []
    for lo, hi in ((0, 1.5), (1.5, 3), (3, 6), (6, 1e9)):
        m = keep & (dist >= lo) & (dist < hi)
        rows.append({'band_km': f'{lo}-{hi}', 'links': int(m.sum()), 'MAE_S1': mean_over(np.abs(p1 - true_all), ev)[m].mean(),
                     'MAE_S3': mean_over(np.abs(p3 - true_all), ev)[m].mean()})
    df = pd.DataFrame(rows); df['change_%'] = 100 * (df.MAE_S3 / df.MAE_S1 - 1)
    df.to_csv(os.path.join(OUTF, 'fig_melb_map_bands.csv'), index=False); print(df.round(2).to_string())
    print('  mean |flow change by T2-3| (veh/15 min), event days %.2f, regular days %.2f' % (np.abs(contrib_ev[inwin]).mean(), np.abs(contrib_reg[inwin]).mean()))
    print('  mean error reduction in window (veh/15 min), event days %.3f, regular days %.3f' % (gain[inwin].mean(), gain_reg[inwin].mean()))


def print_r2():
    """R^2 of the test link flows (all links, entries above one vehicle) for S = 1 and S = 3 (Section 7.3.5 text)."""
    for S in (1, 3):
        if S in Z:
            e, o = Z[S]['test_link_pred'].ravel(), true_all.ravel(); k = o > 1
            e, o = e[k], o[k]
            print(f'  R^2 of the link flows, S = {S}: {1 - ((e - o) ** 2).sum() / ((o - o.mean()) ** 2).sum():.3f}')


for f in (fig_templates, fig_convergence, fig_error_time, fig_vot, fig_attention, fig_map, print_r2):
    f()
