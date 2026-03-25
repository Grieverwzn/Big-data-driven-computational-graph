# Multi-Template Computational Graph (MTCG) for Dynamic Traffic Demand Flow Estimation

This repository contains the code and data for the paper:

> **"Can Contextual Archetypes Explain Daily Traffic Variation? A Multi-Template Computational Graph with Attention-Based Fusion"**
>
> Submitted to *Transportation Research Part B* (Special Issue on "Methodological Advances for Contextual Traffic Management")

## Repository Structure

```
.
├── Sioux Falls/               # Case 2: Sioux Falls network experiments
├── Melbourne/                 # Case 3: Melbourne network experiments
└── Excels/                    # Verification workbooks for Braess and Six-Node networks
    ├── Section42_Verification.xlsx   # Braess network (Section 4.2)
    └── SixNode_Verification.xlsx     # Six-Node network (Section 5.1)
```

---

## Verification Workbooks (Excels/)

### Braess Network — `Section42_Verification.xlsx` (Section 4.2)

A 4-node, 5-link toy network demonstrating the template-based flow decomposition concept.

| Parameter | Value |
|-----------|-------|
| Nodes | 4 |
| Links | 5 |
| OD pairs | 1 (node 1 → node 4) |
| Templates (S) | 3 |
| Paths | 3 |

**Sheets:**
1. **Templates** — Path-link incidence matrix, template flows, verification
2. **Conic Combinations** — Template matrix T, rank check, scenario combinations
3. **Reconstruction Error** — Error analysis for S=2 case

### Six-Node Network — `SixNode_Verification.xlsx` (Section 5.1)

A 6-node, 9-link illustrative network with 4 templates (free-flow, peak congestion, event, incident).

| Parameter | Value |
|-----------|-------|
| Nodes | 6 |
| Links | 9 |
| OD pairs | 1 (node 1 → node 4) |
| Templates (S) | 4 |
| Paths | 5 |

**Sheets:**
1. **Network & Templates** — Incidence matrix, template flows, template matrix T
2. **Conic Combinations** — Observed state flows as conic combinations
3. **Performance Metrics** — RMSE/MAE/MAPE metrics

---

## Case 2: Sioux Falls Network (Section 5.2)

A widely used 24-node benchmark network for evaluating the MTCG under varying numbers of templates.

| Parameter | Value |
|-----------|-------|
| Nodes | 24 |
| Links | 76 |
| OD pairs | 96 |
| Link capacity | 2000 veh/h (uniform) |
| Free-flow travel time | 1.2–4.2 min |
| VDF | BPR: t_a^0 (1 + 0.15 (v/c)^4) |
| Observed links | 66 (86.8%) |
| Unobserved links | 10 (13.2%) |
| Study period | 7:00–9:00 AM |
| Timesteps (T) | 8 (15 min each) |
| Templates tested | S = 1, 2, 3, 4, 5 |
| Candidate paths (K) | 5 per OD pair |
| Training samples | 400 |
| Test samples | 100 |

**Directory structure:**
```
Sioux Falls/Sioux Falls/
├── data/
│   ├── demand.csv              # OD demand (6000 × 96)
│   ├── link_flow.csv           # Link flows (6000 × 76)
│   ├── link_attributes.csv     # Link IDs, nodes, free-flow times, capacity
│   └── od_pair.csv             # 96 OD pairs
├── MTCG-Sioux Falls.ipynb      # Main training notebook
├── Plot.ipynb                  # Visualization notebook
├── Estimations (S=? k=5)/      # Output for each configuration
│   ├── link_flow_estimation.xlsx
│   ├── od_demand_estimation.xlsx
│   ├── loss_train.xlsx
│   ├── loss_test.xlsx
│   └── per_template_results.npz
└── Figures (S=? k=5)/          # Generated figures for each configuration
    ├── demand_flow_pred.png
    ├── sioux_falls_link_flow.png
    ├── sioux_falls_od_demand.png
    ├── sioux_falls_time_dynamics.png
    ├── loss.png
    ├── loss_decomposition.png
    └── ...
```

**Run training:**
```bash
cd "Sioux Falls/Sioux Falls"
# Open MTCG-Sioux Falls.ipynb and set S (number of templates) in the config cell
# Then run all cells
```

**Run visualization:**
```bash
cd "Sioux Falls/Sioux Falls"
# Open Plot.ipynb, set estimation_dir to the desired output folder, then run all cells
```

**Hyperparameters:**
- Optimizer: Adam (beta1=0.5, beta2=0.999, lr=0.002)
- Gradient clipping: max L2 norm = 1
- DNN: 2 hidden layers, 512 neurons, LeakyReLU (slope 0.2)
- Attention dimension: d_k = 128
- Loss weights: mu1=1 (link flow), mu2=0.8 (aggregate OD), mu3=0.01 (VDF)
- Epochs: 1000 (900 for S=2 to avoid divergence)
- Batch size: 64

---

## Case 3: Melbourne Transportation Network (Section 5.3)

A large-scale real-world signalised urban network from Melbourne, Australia.

| Parameter | Value |
|-----------|-------|
| Nodes | 2,077 |
| Links | 4,223 |
| Centroids | 416 |
| OD pairs | 24,160 |
| Virtual links | 3,392 |
| Study period | 6:00–10:00 AM |
| Timesteps (T) | 12 (15 min each) |
| Templates (S) | 3 |
| Candidate paths (K) | 3 per OD pair |

**Directory structure:**
```
Melbourne/Melbourne/
├── data/                          # Raw network data
│   ├── node.csv                   # Node coordinates
│   ├── link.csv                   # Link attributes
│   ├── movement.csv               # Turning movements
│   ├── signal_*.csv               # Signal control data
│   ├── observed_traffic_volume.csv
│   ├── demand_6-10.csv            # Aggregate OD demand
│   └── OD_matrix_*.csv            # 15-min period OD matrices
├── data_1/, data_2/, data_3/      # Preprocessed data per template
│   ├── agent_new.csv              # Vehicle trajectories
│   ├── link_flow.csv              # Link flows
│   ├── OD_pair.csv                # OD pairs
│   └── T*_pred_flow_t*.csv        # Predicted flows per timestep
├── 1-Data-template1 preprocessing.py  # Template 1 data preparation
├── 1-Data-template2 preprocessing.py  # Template 2 data preparation
├── 1-Data-template3 preprocessing.py  # Template 3 data preparation
├── 2-Path generation.py           # K-shortest path generation
├── 3-MTRN.ipynb                   # Model training notebook
├── 4-Plot.ipynb                   # Results visualization
├── Path.ipynb                     # Interactive route map (Folium)
└── Figures/                       # Generated figures
    ├── loss.png
    ├── tend_pred.png
    └── weight.png
```

**Run pipeline:**
```bash
cd "Melbourne/Melbourne"

# Step 1: Preprocess data for each template
python "1-Data-template1 preprocessing.py"
python "1-Data-template2 preprocessing.py"
python "1-Data-template3 preprocessing.py"

# Step 2: Generate candidate path sets
python "2-Path generation.py"

# Step 3: Train MTCG model (open 3-MTRN.ipynb and run all cells)

# Step 4: Generate figures (open 4-Plot.ipynb and run all cells)

# Step 5 (optional): Visualize routes on interactive map (open Path.ipynb)
```

---

## Dependencies

```
numpy
pandas
scipy
torch
matplotlib
seaborn
scikit-learn
networkx
openpyxl
```

**Additional for Melbourne:**
```
pyproj
folium
selenium  # optional, for map PNG export
```

**Hardware:** All experiments were conducted on Intel Core Ultra 5 125H CPU with 32 GB RAM using PyTorch (CPU mode).

---

## Quick Start

```bash
# Clone the repository
git clone <repository-url>

# Install dependencies
pip install numpy pandas scipy torch matplotlib seaborn scikit-learn networkx openpyxl

# Run Sioux Falls experiment (Section 5.2)
cd "Sioux Falls/Sioux Falls"
jupyter notebook MTCG-Sioux\ Falls.ipynb
```

---

## Citation

If you use this code or data, please cite:

```bibtex
@article{wu2026contextual,
  title={Can Contextual Archetypes Explain Daily Traffic Variation? A Multi-Template Computational Graph with Attention-Based Fusion},
  author={Wu, Xin and Shao, Feng},
  journal={Transportation Research Part B},
  year={2026},
  note={Under review}
}
```

---

## Authors

- **Xin (Bruce) Wu**, Department of Civil and Environmental Engineering, Villanova University, USA
- **Feng Shao**, School of Mathematics, China University of Mining and Technology, China

**Contact:** xwu03@villanova.edu

## License

MIT License. Copyright (c) 2026 Xin (Bruce) Wu, Feng Shao. See individual source files for the full license text.
