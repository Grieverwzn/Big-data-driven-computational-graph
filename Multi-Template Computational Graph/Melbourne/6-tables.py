"""Step 6: tables of the Melbourne case (paper Section 7.3, Tables 8-13) and supporting result tables M1-M7.

Purpose
-------
Makes every table of the Melbourne case of the paper from the package's own data (steps 1-4) and MTCG runs (step 5),
plus the supporting tables M1-M7 that the figures of step 7 read.

Paper tables (csv, values rounded as printed in the paper; table numbers of the current manuscript)
  tab_melb_venues.csv      Table 8  (Sec. 7.3.1)  venues, seats / visitors, cars at full size (from data/event_extras.npz)
  tab_melb_data.csv        Table 9  (Sec. 7.3.1)  total OD demand and total sensor counts per time step (mean over the
                                                  100 days, by day type), benchmark VOT (training days)
  tab_melb_candidates.csv  Table 10 (Sec. 7.3.3)  candidate templates T1-T10 and the theta-distance of each one alone
  tab_melb_training.csv    Table 11 (Sec. 7.3.4)  block-wise versus joint training, S = 3 (the joint column needs a
                                                  joint-training run folder: --joint; otherwise only block-wise)
  tab_melb_results.csv     Table 12 (Sec. 7.3.5)  test errors of S = 1, 2, 3 for three link groups and the OD demand;
                           tab_melb_results_by_daygroup.csv: the same by test-day group
  tab_melb_vot.csv         Table 13 (Sec. 7.3.6)  fused VOT per time step and period vs the benchmark VOT, MAPE and
                                                  correlation over the 12 time steps
Three link groups of Tables 11-12 (this replaces the former helper results/recompute_road_errors.py):
  (1) links with sensors    = the 403 sensors used in training, estimated flows vs their counts (5 % sensor error);
  (2) links without sensors = the 45 sensors withheld from training, estimated flows vs their counts;
  (3) all road links        = the 4,223 road links (centroid connectors excluded), estimated vs noise-free flows;
  plus the OD demand of each time step. RMSE and MAE in vehicles (OD: trips per OD pair) per 15 min; MAPE in % over
  the entries whose true value exceeds one vehicle (paper Eqs. 42-44).
Supporting tables M1-M7 (csv, full precision; "all-link" there = all 7,615 links incl. connectors, the convention
of the original experiment logs; the paper tables above use the three link groups):
  M1_errors_overall, M2_errors_by_daytype, M3_vot_by_period, M3b_vot_by_timestep (-> Table 13, Fig. 24a),
  M3c_vot_by_daytype (-> Fig. 24b), M4_benchmark_vot_by_trip_class, M4b_template_vot, M4c_attention_by_daytype and
  M4d_attention_stadium_days (-> Fig. 25), M4e_template_alone_error, M5_blockwise_rounds, M6_blockwise_vs_joint,
  M7_computation.
Definitions: VOT = 60 theta (AUD/h, theta per minute). Fused VOT of day m and step t (paper Eq. 53):
VOT_t^m = 60 sum_s lambda_{s,t}^m theta_{s,t}; reported as plain means over the 80 training days. Benchmark VOT of a
day and step = demand-weighted mean of 60 theta_{w,t} (paper Eq. 52) over the OD pairs with two distinct routes in
the route set of their own trip class (other trips -> T1, stadium trips -> T2, POI trips -> T3).

Usage
-----
    python 6-tables.py                                   (about 1 min)
    python 6-tables.py --results <folder with the run folders> --out <folder> --joint <joint-training S = 3 run>
    e.g. the paper's runs:  python 6-tables.py --results ../Melbourne_r2_v2/results --out reference_results/tables

Inputs
------
    results/mtcg_S{3,2,1}/ (step 5; the long folder names of the original experiment are also recognised),
    data/days100_logit_start7_cap4_k2/ (step 3), data/data_new_{1,2,3}/agent_new.csv (step 2),
    data/event_extras.npz (step 1), data/network/link_xy.csv, OD_pair.csv,
    results/template_selection/selection_single.csv (step 4)
Outputs
-------
    results/tables/tab_melb_*.csv and M*.csv

Authors:
    Xin (Bruce) Wu, Department of Civil and Environmental Engineering, Villanova University, PA, USA
    Feng Shao, School of Mathematics, China University of Mining and Technology, China

Contact: xwu03@villanova.edu (Villanova University), xinwu8592@gmail.com (personal)

MIT License
Copyright (c) 2026 Xin (Bruce) Wu, Feng Shao
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import mtcg_common as C

ap = argparse.ArgumentParser()
ap.add_argument('--results', default=C.RES, help='folder that holds the MTCG run folders')
ap.add_argument('--out', default=os.path.join(C.RES, 'tables'))
ap.add_argument('--selection', default=os.path.join(C.RES, 'template_selection'), help='output folder of step 4')
ap.add_argument('--joint', default='', help='optional: run folder of a joint-training S = 3 run (Table 11, M6)')
args = ap.parse_args()
OUT = args.out; os.makedirs(OUT, exist_ok=True)
W0, T, K, CLOCK = C.W0, C.T, C.K, C.CLOCK
PERIODS = [('whole day (7:00-9:45)', slice(0, 12)), ('7:00-8:30', slice(0, 7)), ('8:45-9:45', slice(7, 12))]
RUNS = [('S3', 3), ('S2', 2), ('S1', 1)]
LABEL = {'S3': 'S=3', 'S2': 'S=2', 'S1': 'S=1'}
DAYTYPE = {'regular': 'regular', 'poi': 'POI', 'stadium': 'stadium', 'both': 'stadium + POI'}

Z = {sp: np.load(os.path.join(C.DAYS, sp, 'days.npz')) for sp in ('train', 'test')}
INFO = {sp: pd.read_csv(os.path.join(C.DAYS, sp, 'days.csv')) for sp in ('train', 'test')}
TYP = INFO['train'].type.values
EVENT = np.isin(TYP, ['stadium', 'both'])                                     # stadium-event days (stadium, both)
dz = Z['train']['od_d']
CLS = np.where(np.isin(dz, list(C.STADIUM_ZONES)), 1, np.where(np.isin(dz, list(C.POI_ZONES)), 2, 0))   # trip class
ROAD = pd.read_csv(os.path.join(C.NETWORK, 'link_xy.csv')).is_connector.values == 0   # 4,223 road links


def alt_routes():
    """OD pairs with two distinct routes among the top-k UE routes of their own template."""
    od = pd.DataFrame({'o_zone_id': Z['train']['od_o'], 'd_zone_id': Z['train']['od_d']})
    alt = np.zeros(len(od), bool)
    for s in range(3):
        a = pd.read_csv(os.path.join(C.template_dir(s + 1), 'agent_new.csv'),
                        usecols=['o_zone_id', 'd_zone_id', 'path_id', 'volume', 'link_sequence'])
        a['volume'] = pd.to_numeric(a['volume'], errors='coerce').fillna(0.0)
        a = a.sort_values(['o_zone_id', 'd_zone_id', 'volume', 'path_id'], ascending=[True, True, False, True])
        a = a[a.groupby(['o_zone_id', 'd_zone_id']).cumcount() < K]
        n = a.groupby(['o_zone_id', 'd_zone_id']).link_sequence.nunique().rename('n').reset_index()
        m = od.merge(n, on=['o_zone_id', 'd_zone_id'], how='left').n.fillna(1).values >= 2
        alt[CLS == s] = m[CLS == s]
    return alt


ALT = alt_routes()


def bench(sp, mask=None):
    """Benchmark VOT (days, T): demand-weighted mean of 60 theta_{w,t} over the OD pairs in mask."""
    q = Z[sp]['demand'][:, W0:W0 + T]; th = Z[sp]['theta_od'][W0:W0 + T]
    if mask is not None:
        q = q[:, :, mask]; th = th[:, mask]
    return 60 * (q * th[None]).sum(2) / np.maximum(q.sum(2), 1e-9)


B_ALL, B_ALT = bench('train'), bench('train', ALT)       # B_ALT = benchmark VOT of the paper (alternative routes)
assert np.allclose(B_ALL, 60 * Z['train']['theta_expected'][:, W0:W0 + T], rtol=1e-4), 'benchmark check'
LA = Z['train']['link_flow_all'][:, W0:W0 + T].transpose(1, 0, 2)                  # (T, days, links), noise-free


def class_bench(c):
    """Benchmark VOT (T,) of trip class c (0 ordinary, 1 stadium, 2 POI), demand-weighted over its OD pairs and days."""
    q = Z['train']['demand'][:, W0:W0 + T][:, :, CLS == c]; th = Z['train']['theta_od'][W0:W0 + T][:, CLS == c]
    qs = q.sum((0, 2))
    return np.where(qs > 0, 60 * (q * th[None]).sum((0, 2)) / np.maximum(qs, 1e-9), np.nan)


def load(R, S):
    """One MTCG run: theta_{s,t} (S, T), attention weights lambda_{s,t}^m of the training days (S, days, T),
    fused VOT (days, T) = 60 sum_s lambda theta (Eq. 53), history, run info, errors, predictions."""
    if R is None or not os.path.exists(os.path.join(R, 'run_info.json')):
        return None
    at = pd.read_csv(os.path.join(R, 'attention_by_day_train.csv'))
    assert (at.day.values.reshape(-1, T)[:, 0] == INFO['train'].day.values).all()
    th = pd.read_csv(os.path.join(R, 'theta_st.csv'), index_col=0).values                         # (S, T)
    w = np.stack([at[f'w_T{s + 1}'].values.reshape(-1, T) for s in range(S)])                    # (S, days, T)
    return dict(R=R, S=S, th=th, w=w, fused=60 * (w * th[:, None, :]).sum(0),
                h=pd.read_csv(os.path.join(R, 'training_history.csv')), ri=json.load(open(os.path.join(R, 'run_info.json'))),
                err=pd.read_csv(os.path.join(R, 'error_table.csv'), index_col=0), p=np.load(os.path.join(R, 'predictions.npz')))


def mape(o, p):
    m = o > 1
    return float(np.mean(np.abs(o[m] - p[m]) / o[m]))


def rmse(o, p):
    return float(np.sqrt(np.mean((o - p) ** 2)))


def errors(r, m=None):
    """Test-day errors for the days in m (M1/M2 convention: all-link = all 7,615 links)."""
    p = r['p']; la = Z['test']['link_flow_all'][:, W0:W0 + T]
    lp = p['test_link_pred'].transpose(1, 0, 2); P = lp[:, :, p['observation_link_number']]
    obs = p['test_link_obs'].transpose(1, 0, 2); oi, ui = p['observed_link_idx'], p['unobserved_link_idx']
    dm, dr = p['test_demand_pred'].transpose(1, 0, 2), p['test_demand_ref'].transpose(1, 0, 2)
    if m is None:
        m = np.ones(la.shape[0], bool)
    return {'sensor MAPE': mape(obs[m][:, :, oi], P[m][:, :, oi]), 'sensor RMSE': rmse(obs[m][:, :, oi], P[m][:, :, oi]),
            'withheld MAPE': mape(obs[m][:, :, ui], P[m][:, :, ui]), 'withheld RMSE': rmse(obs[m][:, :, ui], P[m][:, :, ui]),
            'all-link MAPE': mape(la[m], lp[m]), 'all-link RMSE': rmse(la[m], lp[m]),
            'OD MAPE': mape(dr[m], dm[m]), 'OD RMSE': rmse(dr[m], dm[m])}


def group_errors(r, days):
    """Paper Tables 11-12: RMSE, MAE, MAPE (%) of the three link groups and the OD demand on the test days `days`
    (same computation as the former results/recompute_road_errors.py)."""
    z = r['p']; true_all = Z['test']['link_flow_all'][:, W0:W0 + T].transpose(1, 0, 2)          # (T, days, links)
    sens, oi, ui = z['observation_link_number'], z['observed_link_idx'], z['unobserved_link_idx']
    p = z['test_link_pred']; c = z['test_link_obs']; ps = p[:, :, sens]
    out = {}
    for t, (e, o) in (('Link flow (with sensors)', (ps[:, days][:, :, oi], c[:, days][:, :, oi])),
                      ('Link flow (without sensors)', (ps[:, days][:, :, ui], c[:, days][:, :, ui])),
                      ('Link flow (all road links)', (p[:, days][:, :, ROAD], true_all[:, days][:, :, ROAD])),
                      ('OD demand', (z['test_demand_pred'][:, days], z['test_demand_ref'][:, days]))):
        d = e - o; k = np.abs(o) > 1
        out[t] = (float(np.sqrt(np.mean(d ** 2))), float(np.mean(np.abs(d))), float(100 * np.mean(np.abs(d[k]) / np.abs(o[k]))))
    return out


def alone_road_mape(r):
    """MAPE (%) on all road links of the training days when each template assigns the flows alone (Table 11)."""
    v = r['p']['train_link_pred_per_template']; o = LA[:, :, ROAD]; k = o > 1
    return [100 * float(np.mean(np.abs(v[s][:, :, ROAD] - o)[k] / o[k])) for s in range(v.shape[0])]


def save(df, name, index=False):
    df.to_csv(os.path.join(OUT, name), index=index, encoding='utf-8-sig'); print(f'{name}: {len(df)} rows')


def pct(e):
    return {k: (100 * v if 'MAPE' in k else v) for k, v in e.items()}


# =================================================================================================================
# Paper tables that need only the data and the template selection (steps 1-4)
# =================================================================================================================
# Table 8: venues. Cars at full size = sum of the event trips to each venue zone (2341-2346 stadiums, 2347-2348 POIs)
VENUES = [('Stadium', 'Melbourne Cricket Ground', 100024), ('Stadium', 'AAMI Park', 30050),
          ('Stadium', 'Rod Laver Arena', 14820), ('Stadium', 'Margaret Court Arena', 7500),
          ('Stadium', 'John Cain Arena', 10500), ('Stadium', 'Marvel Stadium', 56000),
          ('POI', 'Queen Victoria Market', 10300), ('POI', 'St Kilda Beach', 3300)]   # as in 1-data_templates.py
ex = np.load(C.EVENT_FILE)
odp = pd.read_csv(os.path.join(C.NETWORK, 'OD_pair.csv'))
cars = (pd.Series(ex['stadium_arrivals']).groupby(odp.d_zone_id.values).sum()
        + pd.Series(ex['poi_total']).groupby(odp.d_zone_id.values).sum())
save(pd.DataFrame([{'Type': ty, 'Venue': v, 'Seats / visitors': n, 'Cars at full size': int(round(cars.loc[2341 + j]))}
                   for j, (ty, v, n) in enumerate(VENUES)]), 'tab_melb_venues.csv')

# Table 9: statistics of the 100 days (OD demand and sensor counts: all 100 days; benchmark VOT: training days)
dem100 = np.concatenate([Z[s]['demand'] for s in ('train', 'test')])[:, W0:W0 + T]
cnt100 = np.concatenate([Z[s]['link_flow'] for s in ('train', 'test')])[:, W0:W0 + T]
typ100 = np.concatenate([INFO[s].type.values for s in ('train', 'test')])
t9 = pd.DataFrame({'Time': CLOCK})
for ty, lab in (('regular', 'Regular'), ('poi', 'POI'), ('stadium', 'Stadium')):
    t9[f'OD demand {lab}'] = (dem100[typ100 == ty].sum(2).mean(0) / 1000).round(1)
for ty, lab in (('regular', 'Regular'), ('poi', 'POI'), ('stadium', 'Stadium')):
    t9[f'Sensor counts {lab}'] = (cnt100[typ100 == ty].sum(2).mean(0) / 1000).round(1)
for ty, lab in (('regular', 'Regular'), ('stadium', 'Stadium')):
    t9[f'Benchmark VOT {lab}'] = B_ALT[TYP == ty].mean(0).round(1)
save(t9, 'tab_melb_data.csv')

# Table 10: candidate templates and the theta-distance of each one alone (reference theta 0.377 = 0.5 x wage / 60)
CAND = [('T1', 'T1', 'Regular day'), ('T2', 'T2', 'Stadium event'), ('T3', 'T3', 'POI event'),
        ('T4', 'lvl080', 'Regular-day demand x0.8'), ('T5', 'lvl120', 'Regular-day demand x1.2'),
        ('T6', 'stad2x', 'Stadium event of double size'), ('T7', 'stad_early', 'Stadium event one hour earlier'),
        ('T8', 'poi_late', 'POI event two hours later'), ('T9', 'other_event', 'Event at two other zones'),
        ('T10', 'both', 'Stadium and POI events on the same day')]          # paper name, name in step 2/4, context
f_single = os.path.join(args.selection, 'selection_single.csv')
if os.path.exists(f_single):
    ss = pd.read_csv(f_single); ss = ss[ss.theta_label == '0.5 x wage'].set_index('candidate')['mean_distance_%']
    save(pd.DataFrame([{'Candidate': p, 'Name in the code': n, 'Context': c, 'theta-distance alone (%)': round(ss[n], 2)}
                       for p, n, c in CAND]), 'tab_melb_candidates.csv')
else:
    print('tab_melb_candidates.csv skipped: run 4-template_selection.py first')

# =================================================================================================================
# Tables from the MTCG runs (step 5)
# =================================================================================================================
runs = {n: load(C.mtcg_run_dir(S, args.results), S) for n, S in RUNS}
missing = [n for n, r in runs.items() if r is None]
if missing:
    print('no results for', missing, '- their rows are left out')
RUNS = [(n, S) for n, S in RUNS if runs[n] is not None]
if not RUNS:
    raise SystemExit('no MTCG results found; run 5-mtcg.py first')

# M1 / M2: errors (experiment convention)
rows = []
for n, S in RUNS:
    r = runs[n]; th, si, ri = r['th'], r['ri']['stop_info'], r['ri']
    rows.append({'run': LABEL[n], **pct(errors(r)), 'epochs': si.get('epochs_run'),
                 'theta at bounds': f'{int(((th < 0.105) | (th > 1.45)).sum())}/{th.size}', 'training min': ri['total_training_time_s'] / 60})
save(pd.DataFrame(rows), 'M1_errors_overall.csv')
typ = INFO['test'].type.values
groups = [('stadium event days (stadium + both)', np.isin(typ, ['stadium', 'both'])), ('POI days', typ == 'poi'),
          ('regular days', typ == 'regular'), ('all test days', np.ones(len(typ), bool))]
save(pd.DataFrame([{'day group': g, 'days': int(m.sum()), 'run': LABEL[n], **pct(errors(runs[n], m))} for g, m in groups for n, _ in RUNS]),
     'M2_errors_by_daytype.csv')

# Table 12: three link groups + OD demand, S = 1, 2, 3 (+ by test-day group)
rows, rows_g = [], []
for n, S in sorted(RUNS, key=lambda x: x[1]):
    for t, (a, b, c) in group_errors(runs[n], np.ones(len(typ), bool)).items():
        nd = 3 if t == 'OD demand' else 2
        rows.append({'Templates': f'S={S}', 'Type of flow': t, 'RMSE': round(a, nd), 'MAE': round(b, nd), 'MAPE (%)': round(c, 2)})
    for g, m in groups:
        for t, (a, b, c) in group_errors(runs[n], m).items():
            rows_g.append({'Templates': f'S={S}', 'day group': g, 'Type of flow': t, 'RMSE': a, 'MAE': b, 'MAPE (%)': c})
save(pd.DataFrame(rows), 'tab_melb_results.csv')
save(pd.DataFrame(rows_g), 'tab_melb_results_by_daygroup.csv')

# M3: VOT
rows = []
for pn, sl in PERIODS + [('whole day without 9:45', slice(0, 11))]:
    row = {'period': pn, 'Benchmark VOT': B_ALT[:, sl].mean(), 'Benchmark VOT, stadium event days': B_ALT[EVENT][:, sl].mean()}
    for n, _ in RUNS:
        row[LABEL[n]] = runs[n]['fused'][:, sl].mean(); row[LABEL[n] + ' bias %'] = 100 * (row[LABEL[n]] / row['Benchmark VOT'] - 1)
    rows.append(row)
m3 = pd.DataFrame(rows); save(m3, 'M3_vot_by_period.csv')
tab = pd.DataFrame({'time': CLOCK, 'Benchmark VOT': B_ALT.mean(0), 'Benchmark VOT, stadium event days': B_ALT[EVENT].mean(0)})
for n, _ in RUNS:
    tab[LABEL[n]] = runs[n]['fused'].mean(0); tab[LABEL[n] + ' bias %'] = 100 * (tab[LABEL[n]] / tab['Benchmark VOT'] - 1)
save(tab, 'M3b_vot_by_timestep.csv')
rows = []
for t in ('regular', 'poi', 'stadium', 'both'):
    sel = TYP == t; row = {'day type': DAYTYPE[t], 'days': int(sel.sum()), 'Benchmark VOT': B_ALT[sel].mean()}
    for n, _ in RUNS:
        row[LABEL[n]] = runs[n]['fused'][sel].mean(); row[LABEL[n] + ' bias %'] = 100 * (row[LABEL[n]] / row['Benchmark VOT'] - 1)
    rows.append(row)
save(pd.DataFrame(rows), 'M3c_vot_by_daytype.csv')

# Table 13: fused VOT by time step and period, bias in parentheses; MAPE and correlation over the 12 time steps
Sv = sorted(S for _, S in RUNS)
fmt = lambda v, b: f'{v:.1f} ({b:+.1f}%)'
rows = [{'Time step / period': tab.time[j], 'Benchmark': round(tab['Benchmark VOT'][j], 1),
         **{f'S={S}': fmt(tab[f'S={S}'][j], tab[f'S={S} bias %'][j]) for S in Sv}} for j in range(T)]
for j, lab in enumerate(('7:00-9:45 (all)', '7:00-8:30', '8:45-9:45 (stadium)')):
    rows.append({'Time step / period': lab, 'Benchmark': round(m3['Benchmark VOT'][j], 1),
                 **{f'S={S}': fmt(m3[f'S={S}'][j], m3[f'S={S} bias %'][j]) for S in Sv}})
b12 = tab['Benchmark VOT'].values
rows.append({'Time step / period': 'MAPE (12 time steps)', 'Benchmark': '--',
             **{f'S={S}': f"{100 * np.mean(np.abs(tab[f'S={S}'].values - b12) / b12):.1f}%" for S in Sv}})
rows.append({'Time step / period': 'Correlation (12 time steps)', 'Benchmark': '--',
             **{f'S={S}': f"{np.corrcoef(tab[f'S={S}'].values, b12)[0, 1]:.2f}" for S in Sv}})
save(pd.DataFrame(rows), 'tab_melb_vot.csv')

# M4: templates
cls = pd.DataFrame({f'Benchmark VOT, {c} trips': class_bench(i) for i, c in enumerate(('ordinary', 'stadium', 'POI'))}, index=CLOCK)
save(cls.rename_axis('time').reset_index(), 'M4_benchmark_vot_by_trip_class.csv')
rows, att, stad, alone = [], [], [], []
m = LA > 1
for n, S in RUNS:
    r = runs[n]; wm = r['w'].mean(1)
    for s in range(S):
        for t in range(T):
            rows.append({'run': LABEL[n], 'template': f'T{s + 1}', 'time': CLOCK[t], 'VOT': 60 * r['th'][s, t], 'theta': r['th'][s, t],
                         'mean weight (training days)': wm[s, t], 'interpretable (weight >= 0.1)': bool(wm[s, t] >= 0.1)})
    v = r['p']['train_link_pred_per_template']; f = r['p']['train_link_pred']
    alone.append({'run': LABEL[n], **{f'T{s + 1} alone all-link MAPE %': 100 * float(np.mean(np.abs(v[s] - LA)[m] / LA[m])) for s in range(S)},
                  'fused all-link MAPE %': 100 * float(np.mean(np.abs(f - LA)[m] / LA[m]))})
    if S > 1:
        att += [{'run': LABEL[n], 'day type': DAYTYPE[t], **{f'T{s + 1}': r['w'][s][TYP == t].mean() for s in range(S)}} for t in ('regular', 'poi', 'stadium', 'both')]
        stad += [{'run': LABEL[n], 'template': f'T{s + 1}', **dict(zip(CLOCK, r['w'][s][EVENT].mean(0)))} for s in range(S)]
save(pd.DataFrame(rows), 'M4b_template_vot.csv')
save(pd.DataFrame(att), 'M4c_attention_by_daytype.csv')
save(pd.DataFrame(stad), 'M4d_attention_stadium_days.csv')
save(pd.DataFrame(alone), 'M4e_template_alone_error.csv')

# M5: block-wise rounds (Section 6: Steps 1-3 form a round; Step 4 = fine-tuning)
rows = []
for n, S in RUNS:
    r = runs[n]; si = r['ri']['stop_info']
    for x in si.get('rounds', []):
        rows.append({'run': LABEL[n], 'round': x['round'], 'fused VOT at round end': x['fused_vot'], 'train link loss': x['train_link'],
                     'theta change vs previous round': x.get('theta_change'), 'link loss drop': x.get('link_drop'), 'weight change': x.get('weight_change'),
                     **{f'T{s + 1} alone sensor MAPE % (train)': 100 * x['train_sensor_mape_alone'][s] for s in range(S)}})
    h = r['h']
    rows.append({'run': LABEL[n], 'round': 'final', 'fused VOT at round end': h.fused_vot.iloc[-1], 'train link loss': h.train_link.iloc[-1],
                 **{f'T{s + 1} alone sensor MAPE % (train)': 100 * h[f'train_sensor_mape_T{s + 1}'].iloc[-1] for s in range(S)}})
    rows.append({'run': LABEL[n], 'round': 'blocks: ' + ', '.join(f'R{b["round"]} {b["block"]} {b["epochs"]}/{b["max"]}' for b in si.get('blocks', []))})
save(pd.DataFrame(rows), 'M5_blockwise_rounds.csv')

# M6 and Table 11: block-wise vs joint training (S = 3); the joint run is not part of this package's pipeline
if 'S3' in runs and runs['S3'] is not None:
    cmp = [('block-wise (r2_v5)', runs['S3'])]
    if args.joint and os.path.exists(os.path.join(args.joint, 'run_info.json')):
        cmp.append(('joint (r2_v4)', load(args.joint, 3)))
    elif args.joint:
        print(f'--joint {args.joint}: no run_info.json there; joint column left out')
    rows, t11 = [], {}
    for lab, r in cmp:
        v = r['p']['train_link_pred_per_template']; ri = r['ri']
        rows.append({'training': lab, **pct(errors(r)), 'fused VOT (whole day)': r['fused'].mean(),
                     'fused VOT bias %': 100 * (r['fused'].mean() / B_ALT.mean() - 1),
                     **{f'T{s + 1} alone all-link MAPE %': 100 * float(np.mean(np.abs(v[s] - LA)[m] / LA[m])) for s in range(3)},
                     **{f'mean weight T{s + 1}': r['w'][s].mean() for s in range(3)},
                     'epochs': ri['stop_info'].get('epochs_run'), 'training min': ri['total_training_time_s'] / 60,
                     'time per epoch s': ri['epoch_time_mean_s'], 'peak memory GB': ri['peak_rss_gb_during_training']})
        e = group_errors(r, np.ones(len(typ), bool)); fv = r['fused'].mean()
        t11['Block-wise' if lab.startswith('block') else 'Joint'] = {
            'MAPE (%) Links with sensors': round(e['Link flow (with sensors)'][2], 2),
            'MAPE (%) Links without sensors': round(e['Link flow (without sensors)'][2], 2),
            'MAPE (%) All road links': round(e['Link flow (all road links)'][2], 2),
            'MAPE (%) OD demand': round(e['OD demand'][2], 2),
            'RMSE Links with sensors': round(e['Link flow (with sensors)'][0], 2),
            'RMSE Links without sensors': round(e['Link flow (without sensors)'][0], 2),
            'RMSE All road links': round(e['Link flow (all road links)'][0], 2),
            f'Fused VOT (AUD/h; benchmark {B_ALT.mean():.1f})': f'{fv:.1f} ({100 * (fv / B_ALT.mean() - 1):+.1f}%)',
            'Mean weights T1/T2/T3': '/'.join(f'{r["w"][s].mean():.2f}' for s in range(3)),
            'Each template alone, MAPE (%) on all road links (training days)': '/'.join(f'{x:.1f}' for x in alone_road_mape(r)),
            'Epochs': ri['stop_info'].get('epochs_run'), 'Training time (min)': round(ri['total_training_time_s'] / 60, 1),
            'Time per epoch (s)': round(ri['epoch_time_mean_s'], 1), 'Peak memory (GB)': round(ri['peak_rss_gb_during_training'], 1)}
    save(pd.DataFrame(rows), 'M6_blockwise_vs_joint.csv')
    save(pd.DataFrame(t11).rename_axis('Item').reset_index(), 'tab_melb_training.csv')

# M7: computation
rows = []
for n, S in RUNS:
    ri = runs[n]['ri']
    rows.append({'run': LABEL[n], 'epochs': ri['stop_info'].get('epochs_run'), 'time per epoch s mean': ri['epoch_time_mean_s'],
                 'time per epoch s sd': ri['epoch_time_sd_s'], 'training s': ri['total_training_time_s'],
                 'inference T=12, 20 test days s mean': ri['inference_time_full_sequence_mean_s'],
                 'inference s sd': ri['inference_time_full_sequence_sd_s'], 'inference per time step s': ri['inference_time_per_step_mean_s'],
                 'peak memory GB': ri['peak_rss_gb_during_training'], 'threads': ri['cpu_count'], 'torch': ri['torch']})
save(pd.DataFrame(rows), 'M7_computation.csv')
