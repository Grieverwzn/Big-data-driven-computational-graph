# Multi-Template Computational Graph (MTCG) for Dynamic Traffic Demand Flow Estimation

This folder contains the code and data of the paper

> **"Can Contextual Archetypes Explain Daily Traffic Variation? A Multi-Template Computational Graph with Attention-Based Fusion"**
>
> Submitted to *Transportation Research Part B* (Special Issue on "Methodological Advances for Contextual Traffic Management")

## Structure

```
Multi-Template Computational Graph/
├── Excels/                  Sections 4.2 and 7.1: verification workbooks (Braess example, Six-Node network)
│                            and the scripts that build them
├── Sioux Falls/SF_r4/       Section 7.2: data generation, templates, MTCG training, baselines,
│                            Tables 5-7 and Figs. 14-19, and the results of the paper's runs
└── Melbourne/Melbourne_r3/  Section 7.3: templates, route sets, data, template selection (SVD, theta-distance),
                             block-wise MTCG training, Tables 8-13 and Figs. 21-26, and the paper's results
```

Each folder has its own README with the steps in order, the requirements, the run times, and which script makes
which table or figure of the paper.

## Quick start

```bash
cd "Sioux Falls/SF_r4"      && python run_all.py     # Sioux Falls (see its README for single steps)
cd "Melbourne/Melbourne_r3" && python run_all.py     # Melbourne (needs path4gmns 0.9.9.post1; see its README)
cd Excels && python create_sixnode_excel.py          # Six-Node workbook (Section 7.1)
```

## Authors

- Xin (Bruce) Wu, Department of Civil and Environmental Engineering, Villanova University, PA, USA
- Feng Shao, School of Mathematics, China University of Mining and Technology, China

Contact: xwu03@villanova.edu (Villanova University), xinwu8592@gmail.com (personal)

MIT License. Copyright (c) 2026 Xin (Bruce) Wu, Feng Shao
