# Sioux Falls (Section 5.2) — data, code and results

Self-contained folder with the data, code and results of the Sioux Falls case study.

```
├── 1-Data generation.py         generates data/link_flow.csv from data/demand.csv
├── 2-MTCG.ipynb                 trains the MTCG                          -> Estimations (S=? k=5)/
├── 3-Plot.ipynb                 draws the figures                        -> Figures (S=? k=5)/
├── 4-Baselines.py               black-box baselines (historical mean, DNN, LSTM, GRU) -> baselines/results/
├── data/                        input data
├── Estimations (S=1…5 k=5)/     main-experiment runs shown in the paper figures
├── Figures (paper)/             the Sioux Falls figures of the paper
├── baselines/results/           baseline results
├── coverage/layouts.csv         sensor layouts of the sensor-coverage experiment
└── tables/                      all results as csv
```

## How to use

**0. Install** Python 3.12 with `torch` (CPU is enough), `numpy`, `pandas`, `networkx`, `matplotlib`, `openpyxl`, `Pillow`, and Jupyter.
Run everything from this folder.

**1. Data (optional).** `data/` is ready to use. To regenerate the link flows from the demand:
```
python "1-Data generation.py"
```

**2. Train the MTCG.** Open `2-MTCG.ipynb`, set the number of templates and the random seed in the config cell
(2nd code cell), then *Run All*:
```python
SEED = 42    # the paper uses 42, 43 and 44
S = 5        # number of templates, 1 to 5
```
Training takes about 1 min (S = 1) to 4 min (S = 5) on a CPU. The results are written to `Estimations (S=5 k=5)/`:
link flows and OD demand (`link_flow_estimation.xlsx`, `od_demand_estimation.xlsx`), each template's flows
(`link_flow_per_template.xlsx`, `per_template_results.npz`), loss curves (`loss_train.xlsx`, `loss_test.xlsx`),
test errors (`error_table.csv`) and the learned θ and attention weights (`theta_attention.npz`).
Running a new S or seed overwrites the folder of that S; copy the provided folders elsewhere first if you want to keep them.

**3. Draw the figures.** Open `3-Plot.ipynb`, set the results folder in the 2nd code cell, then *Run All*:
```python
estimation_dir = "Estimations (S=5 k=5)"
```
The figures are written to `Figures (S=5 k=5)/`. The paper uses `demand_flow_pred.png` (scatter plots), `loss.png` and
`loss_decomposition.png` (loss curves), `sioux_falls_time_dynamics.png` (errors by time step) and
`sioux_falls_link_flow.png` (link flows on the network, last section of the notebook).

**4. Baselines.**
```
python 4-Baselines.py
```
writes the errors of each model and seed to `baselines/results/`.

**5. Sensor coverage.** `coverage/layouts.csv` lists the sensor-equipped links of every layout (column `observed`).
To run one layout, replace the two lines in the cell of `2-MTCG.ipynb` that sets the sensors (the cell starting with
`# Randomly select unobservable link numbers`) by
```python
observed_link = np.array([3, 4, 8, 9, ...])   # the "observed" entry of one row of coverage/layouts.csv
unobserved_link = np.setdiff1d(np.arange(1, num_link + 1), observed_link)
```

**Reproducibility.** The notebook fixes the seed and uses 5 CPU threads (`torch.set_num_threads(5)`), as in the paper runs;
with these settings it reproduces the provided results exactly. A different number of threads gives different results.
The seed sets the initial model parameters and the order of the training mini-batches; all runs use the same data.

## Data

| File | Content |
|---|---|
| `demand.csv` | OD demand, 6000 × 96: 12 time steps × 500 samples (rows ordered by step, then sample); hourly rates (veh/h), one value per 15-min step |
| `link_flow.csv` | link flows, 6000 × 76, same order and unit |
| `link_attributes.csv` | 76 links: start and end node, free-flow travel time (min), capacity (2,000 veh/h) |
| `od_pair.csv` | 96 OD pairs |

- **Demand**: mean about 250 veh/h per OD pair at 7:00, rising to about 340 veh/h at 8:15, plus Gaussian noise with σ = 45 veh/h.
- **Link flows**: recursive Logit loading, with the same structure as the model. At each time step the demand of each OD pair is
  split over its 5 free-flow shortest paths with θ = 0.2; route choice uses the BPR travel times of the previous step
  (free-flow times at 7:00). No sensor error. Over 7:00–8:45 the mean volume-to-capacity ratio is 0.85 (0.96 at 8:15) and
  travel times are 1.15 times the free-flow times on average.
- **Model setting**: first 8 steps (7:00–8:45); samples 1–400 for training, 401–500 for testing; the model receives only
  each sample's total OD demand over the 8 steps and the flows on 66 links; the 10 links without sensors
  (4, 7, 13, 21, 25, 37, 41, 47, 61, 67) are used only for evaluation.

## Results

- `Estimations (S=1 k=5)` … `Estimations (S=5 k=5)`: the main experiment (86.8% sensor coverage), the runs shown in the paper
  figures. Each S was trained with seeds 42, 43 and 44; the folder holds the seed with the lowest training-day sensor MAPE
  (S = 1: 43, S = 2: 42, S = 3: 44, S = 4: 44, S = 5: 43). The numbers in Table 4 of the paper are the mean over the three seeds
  (`tables/T1_table4_mean_sd.csv`); each `Estimations` folder holds one of these seeds, so its `error_table.csv` differs from the mean.
- The recursive demand of the model can give a few small negative OD values (under 1% of the test estimates in the main runs).
  The reported OD demand is max(0, q): the evaluation cell of `2-MTCG.ipynb` sets them to 0, and all OD results and figures use these values.
- `Figures (paper)/`: the Sioux Falls figures of the paper, drawn with `3-Plot.ipynb` from the `Estimations` folders:
  `fig_sf_link_flow_s5.png` (link flows on the network), `fig_scatter_s3.png`, `fig_scatter_s5.png` (estimated vs observed),
  `fig_loss_s5.png`, `fig_loss_decomp_s5.png` (loss curves), `fig_time_dynamics_s1.png`, `fig_time_dynamics_s5.png`
  (errors by time step).
- `tables/`:
  - `T1_table4_mean_sd.csv`, `T1b_table4_per_seed.csv` — MTCG, S = 1–5: mean ± sd over seeds 42–44, and each seed
  - `T2_baselines_mean_sd.csv` — baselines vs MTCG
  - `T3_joint_vs_blockwise.csv` — joint vs block-wise training (seed 42)
  - `T4_coverage_mean_sd.csv`, `T4b_coverage_wins_failures.csv` — sensor coverage (S = 1 and 5, 3 seeds per layout)
  - `T5_computation.csv` — run time and memory
