"""Translate Chinese comments to English and add markdown cells to Melbourne notebooks."""
import json
import re

# Chinese → English translation map for comments
translations = {
    # 3-MTRN.ipynb
    '# <<< 新增：VDF 的 θ，统一设为 2.5，设备/dtype 与 free_flow_tt 保持一致': '# VDF theta parameter, uniformly set to 2.5, device/dtype consistent with free_flow_tt',
    '# ====== 读入三份场景的 demand（转置后形状都是 (16, num_od)）======': '# ====== Load demand from three scenario templates (transposed shape: (16, num_od)) ======',
    '"三份 demand 维度不一致"': '"Demand dimensions inconsistent across three templates"',
    '# 规范化（以防你填的不是严格归一）': '# Normalize weights (in case they do not sum to exactly 1)',
    '# ====== 逐时段加权得到混合 demand ======': '# ====== Compute weighted mixture demand per timestep ======',
    '# 转成你后续用的 torch 张量': '# Convert to torch tensor for downstream use',
    '"demand_6_10.shape =", demand_6_10.shape)  # 应为 (16, num_od)': '"demand_6_10.shape =", demand_6_10.shape)  # Expected: (16, num_od)',
    '# 形状自动广播': '# Shape auto-broadcasts',
    '# MAPE：仅对 |y_true| >= mape_threshold 的样本计算，避免 0/极小值放大': '# MAPE: computed only for |y_true| >= mape_threshold to avoid inflation by near-zero values',
    '# —— 对齐：test_output 的列是"观测列位置索引"，预测是"全网 linkID(0-based)" ——': '# Alignment: test_output columns are observed-column indices; predictions use full-network linkID (0-based)',
    '# 从 DataFrame 的列得到"观测列对应的全网 linkID(0-based)"': '# Get full-network linkID (0-based) corresponding to observed columns',
    "obs_ids_from_df = observation_link_number  # 就是你之前的这一行得到的数组": "obs_ids_from_df = observation_link_number  # Array of observed link IDs",
    '# 位置索引全集（0..num_obs-1）': '# Full set of position indices (0..num_obs-1)',
    '# 各子集的"位置索引"': '# Position indices for each subset',
    '# 将子集位置索引映射到"全网 linkID(0-based)"用于预测张量索引': '# Map subset position indices to full-network linkID (0-based) for prediction tensor indexing',
    '# ———— 取数据并算指标（batch=1 时 squeeze(1) 去掉批次维） ————': '# ---- Extract data and compute metrics (squeeze batch dim when batch=1) ----',
    "thr = 5.0  # MAPE 的真值阈值；想不用过滤就设为 None": "thr = 5.0  # Ground-truth threshold for MAPE; set to None to disable filtering",
    '# Total (所有观测列)': '# Total (all observed columns)',
    '# —— 形成表格并保留变量名 Error（与你原代码兼容） ——': '# ---- Build table and keep variable name Error (backward compatible) ----',
    '# 美化显示': '# Pretty-print',
    '# ====================== 评估指标（替换到此结束） ======================': '# ====================== Evaluation metrics (end of replacement block) ======================',
    '逐小块写 (y_true, y_pred) 到压缩CSV（gz），': 'Write (y_true, y_pred) to compressed CSV (gz) in small chunks,',
    '- 不一次性展开到内存': '- without expanding everything into memory at once',
    '- 只保留 y_true >= thr 的点（避免0附近一大片）': '- keeping only points where y_true >= thr (avoid dense cluster near zero)',
    '- 以 sample_prob 的概率随机抽样（默认2%）': '- randomly sampling with probability sample_prob (default 2%)',
    '# 概率抽样，超简单、超省内存': '# Probabilistic sampling, simple and memory-efficient',
    '# ============== 这里按你的四种配对各自落盘 ==============': '# ============== Write the four estimation pairs to disk ==============',
    '# 1) 全部观测链路': '# 1) All observed links',
    "sample_prob=1,   # 抽2%，按需调大/调小": "sample_prob=1,   # Sample 100%; adjust as needed",
    "thr=5.0             # 只存真值>=5的点；想全存就设 thr=None": "thr=5.0             # Only store points with y_true>=5; set thr=None to store all",
    '# 2) "有传感器"子集': '# 2) "With sensor" subset',
    '# 3) "无传感器"子集': '# 3) "Without sensor" subset',
    '# 4) OD 需求': '# 4) OD demand',

    # 4-Plot.ipynb
    'x轴数据': 'x-axis data',
    '# 绘制折线图': '# Plot line chart',
    '# 设置图例、标题和轴标签': '# Set legend, title, and axis labels',
    '# 调节参数（可选）': '# Adjust parameters (optional)',
    '显示网格线': 'Show grid lines',
    '设置图例字体大小为12': 'Set legend font size',
    '# 显示图形': '# Show figure',
    '# 设置图形大小和字体大小': '# Set figure size and font size',
    '设置Seaborn的绘图风格和字体缩放': 'Set Seaborn plot style and font scaling',
    '# 添加第一个子图': '# Add first subplot',
    '添加竖直线': 'Add vertical line',
    '添加并定位图例': 'Add and position legend',
    '# 添加第二个子图': '# Add second subplot',
    '# 显示图像': '# Show figure',
    '设置默认线条宽度': 'Set default line width',
    '设置坐标轴边框粗细': 'Set axis border width',
    '# 第一个子图': '# First subplot',
    '# 调整x轴和y轴的刻度线粗细': '# Adjust x/y axis tick width',
    '# 第二个子图': '# Second subplot',
    '# 调整子图之间的水平间距': '# Adjust horizontal spacing between subplots',

    # Path.ipynb
    '保存所有满足条件的候选': 'Store all qualifying candidates',
    "raise RuntimeError(\"Top-K需求OD里没找到：3/3/3 且不回环的OD。把 TOP_K 调大或把阈值放宽。\")": 'raise RuntimeError("No qualifying OD found in top-K: need 3/3/3 paths without loops. Increase TOP_K or relax thresholds.")',
    '# 按"更正常(分数小)"优先，其次 demand 大': '# Sort by lower score (more normal) first, then higher demand',
    "# ===== 强制换：选第N名（N从0开始）=====": "# ===== Force selection: pick N-th candidate (0-based) =====",
    "N = 1   # 0=最优，1=第2个，4=第5个……你改这里即可": "N = 1   # 0=best, 1=2nd, 4=5th, etc.",
    '"强制选第"': '"Forced selection of"',
    '"名候选：index1 ="': '"candidate: index1 ="',
    '"候选总数 ="': '"total candidates ="',
    "# 这张表里 x_coord/y_coord 不是 NaN": "# This table has valid x_coord/y_coord (not NaN)",
    '# 定义WGS84坐标系，即EPSG:4326': '# Define WGS84 coordinate system (EPSG:4326)',
    '# 定义墨尔本地区可能使用的UTM区域，例如55S，即EPSG:28355': '# Define UTM zone for Melbourne area, e.g. 55S (EPSG:28355)',
    '# 注意：你可能需要根据实际使用的UTM区域编号来更改这个值': '# Note: adjust the EPSG code based on the actual UTM zone used',
    "utm_zone = Proj(init='epsg:32755')  # 假设55S区域的EPSG代码是28355": "utm_zone = Proj(init='epsg:32755')  # UTM zone 55S",
    '# 看看前几个点（如果有的话）': '# Inspect the first few points',
    '# 定义点坐标，每个点为一个经纬度元组': '# Define point coordinates as (lat, lon) tuples',
    '# 创建起点和终点的标记': '# Create origin and destination markers',
    '# 使用folium.PolyLine创建有向线段，并将其添加到地图上': '# Draw directed polylines on the map using folium.PolyLine',
    '# 保存地图到HTML文件': '# Save map to HTML file',
    '# 1) 先保存 HTML（可选，但建议保留）': '# 1) Save HTML first (optional but recommended)',
    '# 2) 导出 PNG（需要 selenium + 浏览器驱动环境）': '# 2) Export PNG (requires selenium + browser driver)',
    "png_data = map._to_png(5)  # 5 秒等待瓦片加载，可改 2/8/10": "png_data = map._to_png(5)  # 5 seconds to wait for tile loading; adjust as needed",
    '# 原始字符串': '# Original string',
    '# 使用split()方法按分号分割字符串，得到一个包含所有数字的列表': '# Split string by semicolons to get a list of numbers',
    '# 将列表中的每个字符串数字转换为整数': '# Convert each string number in the list to integer',
    '# 使用numpy的array函数将整数列表转换为numpy数组': '# Convert integer list to numpy array',
}

# Markdown cells to insert for each notebook
markdown_cells_mtrn = {
    0: "# MTCG Model Training — Melbourne Network\n\nThis notebook trains the Multi-Template Computational Graph (MTCG) model on the Melbourne transportation network.",
    1: "## 1. Data Loading\n\nLoad network topology, OD pairs, link attributes, and path data for all templates.",
    10: "## 2. Demand Generation\n\nLoad and mix demand from three scenario templates using predefined weights.",
    13: "## 3. Data Preparation\n\nReshape data into tensors and split into training/test sets.",
    19: "## 4. Model Components\n\nDefine the BPR volume-delay function, neural network layers, and attention mechanism.",
    24: "## 5. Model Definition\n\nDefine the MTCG model class with multi-template architecture and attention-based fusion.",
    27: "## 6. Training Setup\n\nConfigure loss functions, optimizer (Adam), and hyperparameters.",
    30: "## 7. Model Training\n\nExecute the training loop with gradient clipping and epoch-wise loss tracking.",
    31: "## 8. Inference & Results\n\nGenerate predictions on the test set and save estimation results.",
    51: "## 9. Loss Export\n\nSave training and test loss trajectories to Excel files.",
    60: "## 10. Evaluation Metrics\n\nCompute RMSE, MAE, and MAPE for link flow (with/without sensors) and OD demand.",
}

markdown_cells_plot = {
    0: "# MTCG Results Visualization — Melbourne Network\n\nThis notebook generates publication figures from the MTCG estimation results.",
    1: "## 1. Loss Curves\n\nPlot training and test loss trajectories over epochs.",
    4: "## 2. Value of Time Distribution\n\nVisualize the estimated value-of-time (VOT) distribution and compare with Melbourne average wage data.",
    7: "## 3. Attention Weights\n\nVisualize template attention weights across timesteps, showing how the model blends templates over time.",
}

markdown_cells_path = {
    0: "# Path Visualization — Melbourne Network\n\nThis notebook selects representative OD pairs and visualizes their template-specific routes on an interactive map.",
    1: "## 1. Data Loading\n\nLoad OD pairs, agent path data, and demand for all three templates.",
    9: "## 2. OD Pair Selection\n\nSelect high-demand OD pairs with diverse, non-looping paths across all templates.",
    14: "## 3. Trajectory Generation\n\nConvert node sequences to geographic coordinates (UTM → WGS84).",
    21: "## 4. Map Visualization\n\nCreate an interactive Folium map with color-coded routes for each template.",
    26: "## 5. Data Export\n\nExport node sequences and path data for further analysis.",
}


def translate_chinese(text):
    """Replace Chinese comments with English translations."""
    for cn, en in translations.items():
        text = text.replace(cn, en)
    return text


def add_markdown_cells(nb, md_cells):
    """Insert markdown cells before specified code cell indices."""
    new_cells = []
    code_idx = 0
    for c in nb['cells']:
        if c['cell_type'] == 'code':
            if code_idx in md_cells:
                md_cell = {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [md_cells[code_idx]]
                }
                new_cells.append(md_cell)
            code_idx += 1
        new_cells.append(c)
    nb['cells'] = new_cells


def process_notebook(filepath, md_cells):
    """Translate Chinese comments and add markdown cells."""
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Translate Chinese in all code cells
    for c in nb['cells']:
        if c['cell_type'] == 'code':
            c['source'] = [translate_chinese(line) for line in c['source']]

    # Add markdown cells
    add_markdown_cells(nb, md_cells)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"Updated {filepath}")


# Process all three notebooks
process_notebook('3-MTRN.ipynb', markdown_cells_mtrn)
process_notebook('4-Plot.ipynb', markdown_cells_plot)
process_notebook('Path.ipynb', markdown_cells_path)

print("\nAll notebooks updated!")
