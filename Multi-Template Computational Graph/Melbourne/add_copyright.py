"""Add copyright and author info to Melbourne notebooks and Python scripts."""
import json
import os
import glob

COPYRIGHT_MD = '''# MTCG: Multi-Template Computational Graph for Traffic Demand Flow Estimation

**Melbourne Network Case Study**

**Authors:**
- Xin (Bruce) Wu, Department of Civil and Environmental Engineering, Villanova University, USA
- Feng Shao, School of Mathematics, China University of Mining and Technology, China

**Contact:** xwu03@villanova.edu

---

MIT License

Copyright (c) 2026 Xin (Bruce) Wu, Feng Shao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.'''

COPYRIGHT_PY = '''\"\"\"
MTCG: Multi-Template Computational Graph for Traffic Demand Flow Estimation
Melbourne Network Case Study

Authors:
    Xin (Bruce) Wu, Department of Civil and Environmental Engineering, Villanova University, USA
    Feng Shao, School of Mathematics, China University of Mining and Technology, China

Contact: xwu03@villanova.edu

MIT License
Copyright (c) 2026 Xin (Bruce) Wu, Feng Shao
\"\"\"
'''

# --- Update notebooks ---
notebook_titles = {
    '3-MTRN.ipynb': 'Model Training',
    '4-Plot.ipynb': 'Results Visualization',
    'Path.ipynb': 'Path Visualization',
}

for nb_file, subtitle in notebook_titles.items():
    filepath = os.path.join(os.path.dirname(__file__), nb_file)
    if not os.path.exists(filepath):
        continue

    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Check if first cell already has copyright
    if nb['cells'] and 'MIT License' in ''.join(nb['cells'][0].get('source', [])):
        print(f'{nb_file}: copyright already present, skipping')
        continue

    # Replace the title in copyright block
    title_md = COPYRIGHT_MD.replace('Melbourne Network Case Study',
                                      f'Melbourne Network — {subtitle}')

    copyright_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [title_md]
    }

    nb['cells'].insert(0, copyright_cell)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f'{nb_file}: copyright added')

# --- Update Python scripts ---
py_scripts = {
    '1-Data-template1 preprocessing.py': 'Template 1 data preprocessing: event-based demand generation',
    '1-Data-template2 preprocessing.py': 'Template 2 data preprocessing: attraction-based demand generation',
    '1-Data-template3 preprocessing.py': 'Template 3 data preprocessing: baseline demand generation',
    '2-Path generation.py': 'K-shortest path generation for all templates',
}

for py_file, description in py_scripts.items():
    filepath = os.path.join(os.path.dirname(__file__), py_file)
    if not os.path.exists(filepath):
        continue

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    if 'MIT License' in content:
        print(f'{py_file}: copyright already present, skipping')
        continue

    header = COPYRIGHT_PY.rstrip() + f'\n# Description: {description}\n\n'
    content = header + content

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f'{py_file}: copyright added')

# --- Also update Others/ scripts ---
others_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'Others')
others_scripts = {
    'create_section42_excel.py': 'Braess network verification (Section 4.2): generates Section42_Verification.xlsx',
    'create_sixnode_excel.py': 'Six-Node network verification (Section 5.1): generates SixNode_Verification.xlsx',
}

COPYRIGHT_PY_GENERAL = '''\"\"\"
MTCG: Multi-Template Computational Graph for Traffic Demand Flow Estimation

Authors:
    Xin (Bruce) Wu, Department of Civil and Environmental Engineering, Villanova University, USA
    Feng Shao, School of Mathematics, China University of Mining and Technology, China

Contact: xwu03@villanova.edu

MIT License
Copyright (c) 2026 Xin (Bruce) Wu, Feng Shao
\"\"\"
'''

for py_file, description in others_scripts.items():
    filepath = os.path.join(others_dir, py_file)
    if not os.path.exists(filepath):
        print(f'{py_file}: not found at {filepath}')
        continue

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    if 'MIT License' in content:
        print(f'{py_file}: copyright already present, skipping')
        continue

    header = COPYRIGHT_PY_GENERAL.rstrip() + f'\n# Description: {description}\n\n'
    content = header + content

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f'{py_file}: copyright added')

print('\nDone!')
