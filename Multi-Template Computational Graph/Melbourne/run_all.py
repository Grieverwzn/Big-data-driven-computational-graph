"""Run the whole Melbourne MTCG pipeline (steps 1-7) in order.

Paper
-----
Section 7.3 (Case 3: Melbourne): all tables (8-13) and figures (21-26) of the Melbourne case.

What
----
1-data_templates.py -> 2-path_generation.py -> 3-data_generation.py -> 4-template_selection.py ->
5-mtcg.py --S 3, 2, 1 -> 6-tables.py -> 7-figures.py. Each step stops the chain if it fails. Steps 2 and 5 can be
restarted: finished path sets are skipped and a killed MTCG run continues from its last checkpoint. Before every MTCG
run the script waits (up to 3 hours, checking every minute) until enough memory is free.

Usage
-----
    python run_all.py                       everything (about 2 h on an 8-core laptop with 16 GB memory)
    python run_all.py --S 1                 only the S = 1 model in step 5 (about 1 h in total)
    python run_all.py --from 5 --S 3,2,1    start at step 5
    python run_all.py --min-free-gb 7       memory needed before each MTCG run (default 7 GB)

Inputs / outputs
----------------
See the numbered scripts and README.md.

Authors:
    Xin (Bruce) Wu, Department of Civil and Environmental Engineering, Villanova University, PA, USA
    Feng Shao, School of Mathematics, China University of Mining and Technology, China

Contact: xwu03@villanova.edu (Villanova University), xinwu8592@gmail.com (personal)

MIT License
Copyright (c) 2026 Xin (Bruce) Wu, Feng Shao
"""
import os
import sys
import time
import argparse
import subprocess
import mtcg_common as C

ap = argparse.ArgumentParser()
ap.add_argument('--from', dest='start', type=int, default=1, help='first step to run (1-7)')
ap.add_argument('--to', dest='stop', type=int, default=7, help='last step to run (1-7)')
ap.add_argument('--S', default='3,2,1', help='MTCG runs of step 5')
ap.add_argument('--threads', type=int, default=8)
ap.add_argument('--min-free-gb', type=float, default=7.0)
ap.add_argument('--ckpt-dir', default=None, help='checkpoint folder of step 5 (default: inside each result folder)')
a = ap.parse_args()
LOG = os.path.join(C.RES, 'run_all.log'); os.makedirs(C.RES, exist_ok=True)


def run(step, script, *extra):
    if not (a.start <= step <= a.stop):
        return
    cmd = [sys.executable, os.path.join(C.ROOT, script), *extra]
    C.log(f'step {step}: {" ".join(cmd[1:])}', LOG)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=C.ROOT)
    if r.returncode != 0:
        C.log(f'step {step} failed (exit code {r.returncode})', LOG); sys.exit(r.returncode)
    C.log(f'step {step} done in {(time.time() - t0) / 60:.1f} min', LOG)


run(1, '1-data_templates.py')
run(2, '2-path_generation.py')
run(3, '3-data_generation.py')
run(4, '4-template_selection.py')
for S in [int(x) for x in a.S.split(',') if x]:
    extra = ['--S', str(S), '--threads', str(a.threads), '--min-free-gb', str(a.min_free_gb)]
    if a.ckpt_dir:
        extra += ['--ckpt-dir', a.ckpt_dir]
    run(5, '5-mtcg.py', *extra)
run(6, '6-tables.py')
run(7, '7-figures.py')
C.log('all steps finished', LOG)
