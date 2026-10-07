# Melbourne (paper Section 7.3) — data, code and results

Code of the Melbourne case of the paper *"Can Contextual Archetypes Explain Daily Traffic Variation? A Multi-Template
Computational Graph with Attention-Based Fusion"* (Section 7.3, Tables 8-13, Figs. 21-26). The scripts rebuild
everything from the open GMNS Melbourne data: the template demand, the route sets, the 100 semi-synthetic days with a
known Logit route choice (benchmark VOT), the number and choice of the templates, the MTCG with S = 1, 2, 3 templates,
and the tables and figures of the paper.

Authors: Xin (Bruce) Wu (Department of Civil and Environmental Engineering, Villanova University, PA, USA) and
Feng Shao (School of Mathematics, China University of Mining and Technology, China).
Contact: xwu03@villanova.edu (Villanova University), xinwu8592@gmail.com (personal). MIT License.

```
├── 1-data_templates.py        template demand and event trips from the open data       -> data/data_new_{1,2,3}/, data/event_extras.npz
├── 2-path_generation.py       route sets by column generation (path4gmns)               -> data/data_new_*/agent_new.csv, data/candidates/
├── 3-data_generation.py       100 semi-synthetic days, Logit benchmark, sensor counts   -> data/days100_logit_start7_cap4_k2/
├── 4-template_selection.py    SVD (number of templates) and theta-distance (choice)     -> results/template_selection/
├── 5-mtcg.py --S 1|2|3        trains and evaluates the MTCG                             -> results/mtcg_S{1,2,3}/
├── 6-tables.py                Tables 8-13 of the paper (+ supporting tables M1-M7)      -> results/tables/
├── 7-figures.py               Figs. 21-26 of the paper                                  -> results/figures/
├── run_all.py                 steps 1-7 in order
├── mtcg_common.py             shared settings (folders, time steps, reading of the route sets)
├── data/open_data/            open GMNS Melbourne data (input)
├── data/network/              renumbered network with centroid connectors (input)
└── reference_results/         the paper's results (tables, run summaries, template selection, Figs. 21-26) for comparison
```

## Requirements

Python 3.10+ with numpy, pandas, scipy, torch (CPU is enough), matplotlib, psutil, pyproj, openpyxl, and
**path4gmns 0.9.9.post1** (`pip install path4gmns==0.9.9.post1`; version 0.10 no longer has
`perform_column_generation`, which step 2 uses). path4gmns is needed only for step 2. Run everything from this folder
(all paths are relative to it); every script prints its options with `-h`.

## How to use

**All at once.**
```bash
python run_all.py                 # steps 1-7 with S = 3, 2, 1 (about 2 hours)
python run_all.py --S 1           # step 5 only for S = 1 (about 1 hour in total)
python run_all.py --from 5        # start at step 5 (steps 1-4 done)
```
Before each MTCG run, `run_all.py` waits (up to 3 h) until enough memory is free (`--min-free-gb`, default 7).

**Step by step.**

| Step | Command | What it does | Time* |
|---|---|---|---|
| 1 | `python 1-data_templates.py` | Template demand from the open data (Sec. 7.3.1): T1 = the observed morning q^0; T2 = q^0 + stadium trips (six stadiums, 12,437 cars at full size, time steps 8:45-9:30 with 10/20/30/40 %); T3 = q^0 + POI trips (two POIs, 7,975 cars, 7:00-7:30 with 10/30/60 %); origins by a gravity rule exp(-3 d / d_max) | 5 s |
| 2 | `python 2-path_generation.py` | Route sets by column generation (path4gmns, 40 iterations, BPR with hourly capacity) for T1-T3 and the seven other candidates T4-T10 of Table 10 | about 17 min (1.5-2 min per template) |
| 3 | `python 3-data_generation.py` | 100 days (80 training / 20 test: regular, POI, stadium, both) by Eq. 51 (OD noise 5 %, event size U(0.5, 1)); link flows by Logit with the known theta_{w,t} of Eq. 52, step by step from 7:00 with BPR times; sensor counts with 5 % error | 30 s, about 3 GB |
| 4 | `python 4-template_selection.py` | SVD of the centred training counts (N_sigma, Sec. 5.1); theta-distance (Sec. 5.2) of T1, T1+T2, T1+T2+T3; greedy, swap and exhaustive choice of 3 of the 10 candidates, at theta^ref = 0.25, 0.5 and 1 x wage | 10 min (3 min with `--thetas 0.5`) |
| 5 | `python 5-mtcg.py --S 1` (then 2, 3) | MTCG (Sec. 4) with block-wise training (Sec. 6) on the 80 training days, evaluation on the 20 test days | S=1 10 min / 4 GB; S=2 25 min / 6 GB; S=3 45 min / 6.5 GB |
| 6 | `python 6-tables.py` | Tables 8-13 and the supporting tables M1-M7 | 30 s |
| 7 | `python 7-figures.py` | Figs. 21-26 | 1 min |

\* Intel Core Ultra 7 265 (20 threads; torch uses 8 by default), 16 GB memory. Steps 2 and 5 can be restarted:
finished route sets are skipped, and a killed MTCG run continues from the start of the block it was in
(`--ckpt-dir` sets where its checkpoint of 1-3 GB is kept; put it outside cloud-synced folders).

## Which script makes which table and figure

| Paper | Content | File | Made by |
|---|---|---|---|
| Table 8 (Sec. 7.3.1) | venues and cars at full size | `tab_melb_venues.csv` | steps 1 -> 6 |
| Table 9 (Sec. 7.3.1) | OD demand, sensor counts, benchmark VOT by time step and day type | `tab_melb_data.csv` | steps 3 -> 6 |
| Table 10 (Sec. 7.3.3) | candidate templates and their theta-distance alone | `tab_melb_candidates.csv` | steps 4 -> 6 |
| Table 11 (Sec. 7.3.4) | block-wise vs joint training, S = 3 | `tab_melb_training.csv` | steps 5 -> 6 (joint column: see below) |
| Table 12 (Sec. 7.3.5) | errors of S = 1, 2, 3 (three link groups, OD demand) | `tab_melb_results.csv` (+ `_by_daygroup`) | steps 5 -> 6 |
| Table 13 (Sec. 7.3.6) | fused VOT vs benchmark VOT by time step and period | `tab_melb_vot.csv` | steps 5 -> 6 |
| Fig. 21 (Sec. 7.3.3) | SVD, swap search, theta-distance by time step | `fig_melb_templates` | steps 3, 4 -> 7 |
| Fig. 22 (Sec. 7.3.4) | block-wise training of S = 3: losses and fused VOT | `fig_melb_convergence` | steps 5 -> 7 |
| Fig. 23 (Sec. 7.3.5) | link-flow RMSE per time step, S = 1, 2, 3 | `fig_melb_error_time` | steps 5 -> 7 |
| Fig. 24 (Sec. 7.3.6) | fused VOT per time step and by day type | `fig_melb_vot` | steps 5, 6 -> 7 |
| Fig. 25 (Sec. 7.3.6) | attention weights of S = 3 | `fig_melb_attention` | steps 5, 6 -> 7 |
| Fig. 26 (Sec. 7.3.6) | inner-city map: error reduction and flow change by Templates 2-3 | `fig_melb_map` | steps 5 -> 7 |

Tables are written to `results/tables/`, figures (png + pdf) to `results/figures/`. Fig. 20 (the network) is a
picture, not made by the code. The errors of Tables 11-12 use three link groups: (1) links with sensors = the 403
sensors used in training, against their counts; (2) links without sensors = the 45 sensors withheld from training,
against their counts; (3) all road links = the 4,223 road links (centroid connectors excluded), against the noise-free
synthesized flows; RMSE and MAE in vehicles per 15 min, MAPE over the values above one vehicle. The supporting tables
M1-M2 keep the convention of the experiment logs (all 7,615 links including connectors).

The joint-training column of Table 11 comes from a separate run in which all parameter blocks are updated together
from the start (not part of this pipeline); give its run folder with `python 6-tables.py --joint <folder>`. Without it,
Table 11 has only the block-wise column; the paper's values of both columns are in `reference_results/tables/`.

## Data

| Folder / file | Content |
|---|---|
| `data/open_data/` (input, 5.9 MB) | open GMNS Melbourne data: OD matrices of the 16 slots 6:00-10:00 (15 min), nodes, links, counts of 448 sensors |
| `data/network/` (input, 3.3 MB) | renumbered network with centroid connectors (7,615 links of which 4,223 road links, 2,348 nodes, 30,040 OD pairs), link and node coordinates for the map |
| `data/data_new_{1,2,3}/` (step 1-2, 36 MB each) | template demand `demand_6-10.csv` and route sets `agent_new.csv` of T1-T3 |
| `data/candidates/` (step 2, 252 MB) | demand and route sets of the candidates T4-T10 |
| `data/pathgen/` (step 2, 47 MB) | path4gmns working folders (logs) |
| `data/event_extras.npz` (step 1) | stadium and POI event trips per OD pair |
| `data/days100_logit_start7_cap4_k2/` (step 3, 63 MB) | the 100 days: `train/` and `test/` with `days.npz` (demand, sensor counts, noise-free flows of all links, theta_{w,t}) and `days.csv` (day type, event sizes) |

`data/network/` was made from the open data by the preprocessing of the first version of the paper (centroid
connectors to the four nearest nodes, renumbering of nodes and links, the eight venue zones 2341-2348 added); it is
included as input. The generated data (about 0.5 GB) and the MTCG results (0.2-0.4 GB per run) are not stored in the
repository; steps 1-5 regenerate them.

- **Days**: 50 regular days, 25 POI days, 15 stadium days and 10 days with both events; 80 training days and 20 test
  days with the same proportions. Each day covers the 12 time steps 7:00-9:45.
- **Benchmark VOT**: theta_{w,t} = 0.377 x peak factor (1.2 at 7:30-8:15) x trip factor (stadium 1.3, POI 1.1) x OD
  factor exp(N(0, 0.15^2)) per minute; VOT = 60 theta AUD/h. The model sees only the period OD demand of each day and
  the counts of the 403 training sensors.

## Results and reproducibility

`reference_results/` holds the paper's results: the tables (`tables/tab_melb_*.csv` = Tables 8-13, and M1-M7), the
small files of the three MTCG runs (`mtcg_S{1,2,3}/`: error table, theta, attention weights, training history, run
information), the template selection (`template_selection/`), and the six paper figures (`figures/fig_melb_*.pdf`,
Figs. 21-26). To make the tables and figures from other run folders:
`python 6-tables.py --results <folder> --out <folder>/tables` and
`python 7-figures.py --results <folder> --tables <folder>/tables --out <folder>/figures`.

The package was checked against the files of the paper's experiment:
- steps 1-3 reproduce the template demand, the event trips, the ten route sets and all arrays of the 100 days
  bit for bit;
- step 4 reproduces the SVD shares (38.2 %, 22.3 %; N_sigma = 2) and the theta-distances (T1 alone 5.31 %, T1+T2
  5.17 %, T1+T2+T3 5.15 %, rank 1 of 120 sets) and all its csv files;
- steps 6-7 applied to the paper's MTCG runs give Tables 8-13 with the values printed in the paper and Figs. 21-26
  pixel-identical to the paper's figures;
- step 5 is the training code of the paper's notebooks (`3-MTCG-r2v5_S{1,2,3}`) turned into a script with the same
  settings and seeds. Torch used 20 CPU threads in the paper's runs and uses 8 here by default (`--threads`), so a new
  run can differ slightly in the last digits. Paper values (Table 12, 20 test days, MAPE % / RMSE):

  | run | links with sensors | links without sensors | all road links | OD demand |
  |---|---|---|---|---|
  | S = 1 | 5.25 / 17.22 | 5.81 / 18.40 | 2.93 / 7.41 | 5.60 / 0.237 |
  | S = 2 | 5.12 / 16.54 | 6.26 / 16.72 | 2.97 / 6.25 | 6.22 / 0.237 |
  | S = 3 | 5.30 / 16.40 | 6.47 / 16.37 | 3.27 / 6.29 | 6.41 / 0.238 |

## Authors and license

Authors:
- Xin (Bruce) Wu, Department of Civil and Environmental Engineering, Villanova University, PA, USA
- Feng Shao, School of Mathematics, China University of Mining and Technology, China

Contact: xwu03@villanova.edu (Villanova University), xinwu8592@gmail.com (personal)

MIT License
Copyright (c) 2026 Xin (Bruce) Wu, Feng Shao
