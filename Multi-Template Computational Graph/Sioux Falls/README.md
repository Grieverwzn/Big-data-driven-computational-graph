# Sioux Falls (Section 7.2) — data, code and results

Self-contained folder with the data, code and results of the Sioux Falls case study of the paper
"Can Contextual Archetypes Explain Daily Traffic Variation? A Multi-Template Computational Graph with Attention-Based
Fusion" (Section 7.2: Tables 5–7, Figs. 14–19). All scripts use paths relative to this folder.

```
├── 1-data_generation.py       generates the 500 samples (OD demand, link flows)          -> data/
├── 2-template_generation.py   builds the route-set templates (k-means, Sec. 7.2.2)       -> templates/
├── 3-mtcg.py                  trains the MTCG once (one S, one seed)                     -> results/S<S>_seed<seed>[_odw]/
├── 4-baselines.py             baselines of Table 6 (historical mean, DNN, LSTM, GRU, DNN-path, GCN) -> results/baselines/
├── 5-tables_and_figures.py    Tables 5–7 and Figs. 14–19                                 -> tables/, figures/
├── run_all.py                 runs steps 1–5 in order (restartable)
├── sf_common.py               shared helpers (network, paths, BPR, folders); imported by the scripts
├── data/                      network, generated data, sensor layouts of Table 7
├── templates/                 templates for S = 1..5
├── results/                   the paper's runs (all seeds), baselines and coverage runs
├── tables/                    Tables 5–7 (csv and LaTeX body)
└── figures/                   Figs. 14–19 as in the paper
```

## How to use

**0. Install.** Python 3 with `numpy`, `pandas`, `networkx`, `scikit-learn`, `torch` (CPU is
enough), `matplotlib`, `Pillow` and `openpyxl` (tested with Python 3.12.10, torch 2.10.0+cpu, numpy 2.4.6,
pandas 2.3.3, networkx 3.6.1, scikit-learn 1.8.0, matplotlib 3.11.0). The figures use the font
Times New Roman (matplotlib falls back to DejaVu Serif if it is missing; the figures then differ slightly from the paper).
Run everything from this folder.

**1. Data.** `data/` is ready to use. To regenerate it (fixed seed 42, identical output on every run, about 2 s):
```
python "1-data_generation.py"
```
It reads only `data/link_attributes.csv` and `data/od_pair.csv` and writes the other files of `data/` (see *Data* below).

**2. Templates.** `templates/` is ready to use. To regenerate it (deterministic, about 10 s):
```
python "2-template_generation.py"
```
Template 1 holds the 5 free-flow shortest paths of every OD pair. For templates 2..S, k-means groups the 400 training
samples into S − 1 groups by their aggregate OD demand; the mean of each group divided by T = 8 is assigned to the
free-flow paths (Logit, θ = 0.2, BPR, method of successive averages), and under these link times each template keeps the
3 free-flow shortest paths and adds the 2 shortest paths not among them (paper Sec. 7.2.2).

**3. Train the MTCG** (one run = one number of templates S and one seed):
```
python "3-mtcg.py" --S 5 --seed 44                    # standard setting (Table 6, rows MTCG S = 1..5)
python "3-mtcg.py" --S 5 --seed 43 --od-weighted      # OD-weighted setting, mu_2 = 5 and decaying learning rate
python "3-mtcg.py" --S 1 --seed 42 --layout n23_L0 --no-test-loss   # one sensor-coverage run (Table 7)
python "3-mtcg.py" --S 1 --seed 42 --iters 5          # quick check that everything runs (a few seconds)
```
Settings as in the paper (Sec. 7.2.2): Adam (learning rate 0.001, decay rates 0.9 / 0.999), mini-batches of 64 of the
400 training samples, 1000 iterations, gradient clipping at norm 1, DNN layers 512, attention size d_k = 128, loss
weights mu_1 = 1, mu_2 = 0.8, mu_3 = 0.01, mu_4 = 0. Options: `--threads` (default 8), `--iters` (default 1000).
Output in `results/S<S>_seed<seed>[_odw]/` (coverage runs: `results/coverage/<layout>_s<seed>_S<S>/`):

| File | Content |
|---|---|
| `link_flow_estimation.xlsx` | test link flows, columns Estimation / Observation (8 steps × 100 samples × 76 links, step-major) |
| `od_demand_estimation.xlsx` | test OD demand, columns Estimation / Reference (8 × 100 × 96); estimates below 0 are set to 0 |
| `loss_train.xlsx`, `loss_test.xlsx` | losses at every iteration: link_flow (L_V), demand (L_Q), vdf (L_D), total |
| `theta_attention.npz` | \|θ\| (S × 8 × 96), test attention weights (8 × 100 × S), links with / without sensors |
| `error_table.csv` | test RMSE, MAE (veh/h) and MAPE of links with sensors, without sensors, all links, OD demand |
| `run_info.json` | settings, error table, training-sample sensor MAPE, θ and attention summaries, run time |

**4. Baselines** (Table 6):
```
python "4-baselines.py"                               # all models, seeds 42, 43, 44
python "4-baselines.py" --models dnnpath,gcn --seeds 42
```
Same input, supervision (66 links with sensors), test samples and error formulas as the MTCG. The learning rate of each
model is chosen from {0.001, 0.002, 0.005} on validation samples 350–399. Output in `results/baselines/`: `runs.csv`
(per model, seed and flow type), `info.json` (learning rates, run times), `summary_mean_sd.csv`.

**5. Tables and figures:**
```
python "5-tables_and_figures.py"
```
reads `data/` and `results/` and writes `tables/` and `figures/` (about 1–2 min). For the figures of one setting the run
with the lowest training-sample sensor MAPE is drawn (no test data are used to choose it): S = 1 seed 43, S = 5 seed 44,
S = 5 OD-weighted seed 43.

**All steps at once:**
```
python run_all.py                  # steps 1-5; add --coverage for the 120 coverage runs of Table 7
```
A training run whose `run_info.json` exists is skipped, and the baselines are skipped when `results/baselines/runs.csv`
exists. Because `results/` ships with the paper's runs, `run_all.py` then only regenerates the data, the templates, the
tables and the figures. To repeat all experiments, delete or rename `results/` first.

## Which script makes which table and figure

| Paper | Content | File | Script |
|---|---|---|---|
| Table 5 (Sec. 7.2.1) | mean OD demand, v/c and travel-time ratio by time step | `tables/table5.tex`, `.csv` | 1 → 5 |
| Fig. 14 (Sec. 7.2.1) | trips leaving / arriving at each node | `figures/fig_sf_od_nodes.png` | 1 → 5 |
| Fig. 15 (Sec. 7.2.2) | stacked SVD of the training link flows, N_σ = 3 | `figures/fig_sf_svd.png` | 1 → 5 |
| Table 6 (Sec. 7.2.3) | baselines, MTCG S = 1..5, MTCG S = 5 OD-weighted (mean ± sd over seeds 42–44) | `tables/table6.tex`, `table6_mean_sd.csv`, `table6_runs.csv` | 3, 4 → 5 |
| Fig. 16 (Sec. 7.2.3) | estimates over the time steps, S = 5 | `figures/fig_time_dynamics_s5_v2.jpg` | 3 → 5 |
| Fig. 17 (Sec. 7.2.4) | per-link MAPE (S = 1, S = 5), MAPE and attention by day type | `figures/fig_sf_link_error_D.jpg` | 3 → 5 |
| Fig. 18 (Sec. 7.2.4) | estimated vs observed link flows and OD trips, (a) S = 1, (b) S = 5 | `figures/fig_scatter_s1_new.jpg`, `fig_scatter_s5_new.jpg` | 3 → 5 |
| Fig. 19 (Sec. 7.2.4) | training and test losses, OD-weighted S = 5 | `figures/fig_sf_convergence_od_weighted.jpg` | 3 → 5 |
| Table 7 (Sec. 7.2.5) | sensor coverage 30.3–86.8%, S = 1 vs S = 5 (mean ± sd over 5 layouts) | `tables/table7.tex`, `table7_mean_sd.csv`, `table7_runs.csv` | 3 (`--layout`) → 5 |

Fig. 13 (the network) is not generated by code. The figures in `figures/` are identical (pixel by pixel) to the files used
in the paper, and the LaTeX bodies of Tables 6 and 7 are identical to the paper's tables.

## Data

| File in `data/` | Content |
|---|---|
| `link_attributes.csv`, `od_pair.csv` | network (24 nodes, 76 links, capacity 2,000 veh/h) and the 96 OD pairs (input of step 1) |
| `demand.csv` | OD demand q^m_{w,t}, veh/h per 15-min step; 6000 rows = 12 steps × 500 samples (step-major), 96 columns |
| `link_flow.csv` | link flows, veh/h; 6000 rows (same order), 76 columns |
| `day_info.csv` | day type, time shift ω^m (steps) and θ^m of each sample |
| `paths_template1.npz` | the 5 free-flow shortest paths per OD pair (link × path matrix) |
| `paths_type.npz` | route sets and OD shares of the three day types |
| `checks.json` | basic statistics of Sec. 7.2.1 (mean demand by step, CV, v/c, travel-time ratio, N_σ) |
| `sensor_layouts.csv` | the 20 random sensor layouts of Table 7 (23, 38, 53, 66 links with sensors; 5 layouts each) |

Samples 1–400 are used for training and 401–500 for testing. Each sample draws one of three day types (spatial
distributions of the OD trips, each with its own route set), shifts the designed time-dependent demand by up to 7.5 min,
and draws its Logit parameter θ^m from U(0.18, 0.22); its link flows come from one forward pass of the three-layer
computational graph at each 15-min step (BPR travel times of the previous step). The model uses only the aggregate OD
demand of each sample (sum over the 8 steps 7:00–8:45) and the flows on the 66 links with sensors; the 10 links without
sensors (4, 7, 13, 21, 25, 37, 41, 47, 61, 67) and the per-step demand are used only to evaluate the estimates.

## Results

- `results/S<S>_seed<seed>/` (S = 1..5, seeds 42, 43, 44) and `results/S5_seed<seed>_odw/` (OD-weighted): the runs of
  Table 6. To keep the folder small, the test estimates (`link_flow_estimation.xlsx`, `od_demand_estimation.xlsx`) are
  kept only for the two runs drawn in Figs. 16–18 (`S1_seed43`, `S5_seed44`); the other runs keep `run_info.json`,
  `error_table.csv`, the losses and `theta_attention.npz`. Rerun `3-mtcg.py` with the same S and seed to get the full output.
- `results/baselines/`: the baseline runs of Table 6.
- `results/coverage/<layout>_s<seed>_S<S>/`: the 120 coverage runs of Table 7 (`run_info.json`, `error_table.csv` only).
  For each layout and S, the seed with the lowest training-sample sensor MAPE is kept (`tables/table7_runs.csv`, column `kept`).
- The recursive demand of the model can give a few small negative OD values; the reported OD demand is max(0, q).

## Run time and reproducibility

Run time on a laptop CPU with 8 threads: data about 2 s, templates about 10 s, one MTCG run about 1 min (S = 1) to
4–9 min (S = 5), all 18 runs of Table 6 about 1.5 h, baselines 30–60 min, the 120 coverage runs about 4–6 h, tables and
figures 1–2 min. Memory below 2 GB.

Each run fixes its random seed (initial parameters and order of the mini-batches; the data are the same for all runs).
With the same number of CPU threads the results are reproduced exactly: rerunning `3-mtcg.py` with this package
reproduced the paper runs S = 1 seed 42 (`--threads 5`), S = 5 OD-weighted seed 43, and the coverage runs n23_L0 S = 1
seed 42 and S = 5 seed 44 (`--threads 8`) to the last digit. A different number of threads can change the last digits of
floating-point sums and, through the 1000 iterations, the results slightly; this affects the trained baselines too (GCN
most: its test MAPE on links without sensors varied between 7.2% and 7.5% for seed 42 with 1–20 threads).

Note on `templates/lp_S1.npz`: the original MTCG notebook gave the free-flow times to the path search in single
precision. Several paths have equal free-flow length, and with single precision the ties are broken differently for 23
of the 96 OD pairs (9 with a different set of paths) than in template 1 of `lp_S2`..`lp_S5`. Both are valid free-flow
shortest-path sets; `lp_S1.npz` keeps the notebook's version so that the S = 1 results are reproduced.

## Authors and license

Authors:
- Xin (Bruce) Wu, Department of Civil and Environmental Engineering, Villanova University, PA, USA
- Feng Shao, School of Mathematics, China University of Mining and Technology, China

Contact: xwu03@villanova.edu (Villanova University), xinwu8592@gmail.com (personal)

MIT License  
Copyright (c) 2026 Xin (Bruce) Wu, Feng Shao
