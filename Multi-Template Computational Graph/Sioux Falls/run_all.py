"""
Run the whole Sioux Falls pipeline of the Multi-Template Computational Graph (MTCG), paper "Can Contextual Archetypes
Explain Daily Traffic Variation? A Multi-Template Computational Graph with Attention-Based Fusion" (Section 7.2:
Tables 5-7, Figs. 14-19).

Authors:
    Xin (Bruce) Wu, Department of Civil and Environmental Engineering, Villanova University, PA, USA
    Feng Shao, School of Mathematics, China University of Mining and Technology, China

Contact: xwu03@villanova.edu (Villanova University), xinwu8592@gmail.com (personal)

MIT License
Copyright (c) 2026 Xin (Bruce) Wu, Feng Shao

Usage  : python run_all.py [--threads 8] [--skip-data] [--skip-baselines] [--coverage]
         Runs, one process at a time and in this order:
           1-data_generation.py      data/ (Sec. 7.2.1); skipped with --skip-data (the output is identical on every run)
           2-template_generation.py  templates/ (Sec. 7.2.2)
           3-mtcg.py                 S = 1..5 with seeds 42, 43, 44 (standard setting), then S = 5 OD-weighted with
                                     seeds 42, 43, 44 (18 runs, Table 6)
           3-mtcg.py --layout        only with --coverage: 20 sensor layouts x seeds 42, 43, 44 x S = 1, 5 (120 runs,
                                     Table 7, Sec. 7.2.5; without the per-iteration test loss)
           4-baselines.py            baselines of Table 6; skipped with --skip-baselines or when
                                     results/baselines/runs.csv exists (delete it to rerun the baselines)
           5-tables_and_figures.py   tables/ and figures/
         A training run whose run_info.json exists in results/ is skipped, so an interrupted pipeline can be restarted.
         The package ships with the paper's runs in results/, so this script then only redraws the tables and figures;
         delete (or rename) results/ to repeat all experiments.
         Run time on a laptop CPU with 8 threads: about 2-3 hours without --coverage (MTCG runs about 1.5 h, baselines
         30-60 min); the 120 coverage runs add about 4-6 hours.
"""
import os, sys, subprocess, argparse, time
import pandas as pd
from sf_common import HERE, DATA, RESULTS, SEEDS, run_dir

ap = argparse.ArgumentParser(description='Run the Sioux Falls pipeline')
ap.add_argument('--threads', type=int, default=8, help='torch CPU threads of each training run')
ap.add_argument('--skip-data', action='store_true', help='do not regenerate data/')
ap.add_argument('--skip-baselines', action='store_true', help='do not run 4-baselines.py')
ap.add_argument('--coverage', action='store_true', help='also run the 120 sensor-coverage runs of Table 7')
args = ap.parse_args()


def run(script, *a, retries=1):
    """run one script of the package in a new Python process (one retry on failure)"""
    for attempt in range(retries + 1):
        t0 = time.time()
        print(f'>>> {script} {" ".join(a)}', flush=True)
        r = subprocess.run([sys.executable, '-W', 'ignore', os.path.join(HERE, script), *a], cwd=HERE)
        if r.returncode == 0:
            print(f'<<< done in {time.time() - t0:.0f} s', flush=True)
            return
        print(f'<<< FAILED (exit code {r.returncode})', flush=True)
    sys.exit(f'stopped: {script} failed')


if not args.skip_data:
    run('1-data_generation.py')
run('2-template_generation.py')

# MTCG runs of Table 6: (S, seed, OD-weighted)
jobs = [(S, seed, False) for S in range(1, 6) for seed in SEEDS] + [(5, seed, True) for seed in SEEDS]
for S, seed, odw in jobs:
    if os.path.exists(os.path.join(run_dir(S, seed, odw), 'run_info.json')):
        print(f'skip S={S} seed={seed}{" OD-weighted" if odw else ""} (finished before)')
        continue
    run('3-mtcg.py', '--S', str(S), '--seed', str(seed), '--threads', str(args.threads), *(['--od-weighted'] if odw else []))

# sensor-coverage runs of Table 7 (optional)
if args.coverage:
    for layout in pd.read_csv(os.path.join(DATA, 'sensor_layouts.csv'))['name']:
        for S in (1, 5):
            for seed in SEEDS:
                if os.path.exists(os.path.join(run_dir(S, seed, layout=layout), 'run_info.json')):
                    print(f'skip coverage {layout} S={S} seed={seed} (finished before)')
                    continue
                run('3-mtcg.py', '--S', str(S), '--seed', str(seed), '--layout', layout, '--no-test-loss',
                    '--threads', str(args.threads))

if args.skip_baselines:
    pass
elif os.path.exists(os.path.join(RESULTS, 'baselines', 'runs.csv')):
    print('skip 4-baselines.py (results/baselines/runs.csv exists; delete it to rerun the baselines)')
else:
    run('4-baselines.py', '--threads', str(min(args.threads, 4)))
run('5-tables_and_figures.py')
