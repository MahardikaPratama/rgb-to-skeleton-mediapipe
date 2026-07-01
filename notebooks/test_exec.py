import pickle
import numpy as np
import pandas as pd
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import wilcoxon
import statsmodels.stats.multitest as smm
import itertools
import warnings
warnings.filterwarnings('ignore')

PICKLE_FILE = '../data/pickle/pose_bisindo.pkl'
OUTPUT_DIR = '../data/results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(PICKLE_FILE, 'rb') as f:
    data = pickle.load(f)

video_ids = list(data.keys())
speakers = sorted(set(v.split('_')[0] for v in video_ids))
speakers = [s for s in speakers if s in ['P1', 'P2', 'P3', 'P4', 'P5']]
signer_labels = [f"P0{s[1:]}" for s in speakers]

WRIST_LH = 0
WRIST_RH = 21

print("Data loaded successfully.")

# ── 1. Ekstraksi Centroid Tingkat Video ─────────────────────────────────────
centroids = []

for vid in video_ids:
    sp = vid.split('_')[0]
    if sp not in speakers:
        continue
    sentence = "_".join(vid.split('_')[1:]) # S01_R1
        
    kp = data[vid]['keypoints']
    
    # Left Wrist
    w_l = kp[:, WRIST_LH, :]
    valid_l = ~( (w_l[:, 0] == 0) & (w_l[:, 1] == 0) )
    if valid_l.sum() > 0:
        lh_x = np.mean(w_l[valid_l, 0])
        lh_y = np.mean(w_l[valid_l, 1])
    else:
        lh_x, lh_y = np.nan, np.nan
        
    # Right Wrist
    w_r = kp[:, WRIST_RH, :]
    valid_r = ~( (w_r[:, 0] == 0) & (w_r[:, 1] == 0) )
    if valid_r.sum() > 0:
        rh_x = np.mean(w_r[valid_r, 0])
        rh_y = np.mean(w_r[valid_r, 1])
    else:
        rh_x, rh_y = np.nan, np.nan
        
    centroids.append({
        'vid': vid,
        'signer': sp,
        'sentence': sentence,
        'lh_x': lh_x, 'lh_y': lh_y,
        'rh_x': rh_x, 'rh_y': rh_y
    })

df = pd.DataFrame(centroids)
print(f"Centroids extracted for {len(df)} videos.")

# ── 2. Gambar 1: Boxplot Distribusi Posisi ──────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'axes.edgecolor': '#333333',
    'axes.linewidth': 0.8,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})

fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))

panels = [
    {'ax': axes[0,0], 'var': 'lh_x', 'title': '(a) Tangan Kiri — Koordinat X (Horizontal)', 'color': '#AEC7E8', 'line_c': '#1F77B4'},
    {'ax': axes[0,1], 'var': 'lh_y', 'title': '(b) Tangan Kiri — Koordinat Y (Vertikal)',   'color': '#FF9896', 'line_c': '#D62728'},
    {'ax': axes[1,0], 'var': 'rh_x', 'title': '(c) Tangan Kanan — Koordinat X (Horizontal)', 'color': '#98DF8A', 'line_c': '#2CA02C'},
    {'ax': axes[1,1], 'var': 'rh_y', 'title': '(d) Tangan Kanan — Koordinat Y (Vertikal)',   'color': '#FFBB78', 'line_c': '#FF7F0E'},
]

for p in panels:
    ax = p['ax']
    var = p['var']
    
    data_list = [df[df['signer'] == sp][var].dropna().values for sp in speakers]
    ds_median = df[var].median()
    
    ax.axhline(ds_median, color=p['line_c'], linestyle='--', linewidth=1.4, zorder=1, label=f'Median dataset = {ds_median:.3f}'.replace('.', ','))
    
    bp = ax.boxplot(data_list, positions=range(len(speakers)), patch_artist=True, widths=0.45, zorder=3,
                    medianprops=dict(color=p['line_c'], linewidth=2.5),
                    boxprops=dict(facecolor=p['color'], edgecolor='#555555', linewidth=1.0),
                    whiskerprops=dict(color='#555555', linewidth=1.2),
                    capprops=dict(color='#555555', linewidth=1.2),
                    showfliers=False)
    
    for i, sp_data in enumerate(data_list):
        x_jitter = np.random.normal(i, 0.04, size=len(sp_data))
        ax.scatter(x_jitter, sp_data, color='#888888', s=12, alpha=0.5, zorder=2)
        
        sp_med = np.median(sp_data)
        ax.text(i, sp_med + 0.015, f"{sp_med:.3f}".replace('.', ','), ha='center', va='bottom', 
                color=p['line_c'], fontweight='bold', fontsize=10)
    
    medians = [np.median(d) for d in data_list]
    rentang = max(medians) - min(medians)
    textstr = f"Rentang antar penutur\n(maks-min median) = {rentang:.3f}".replace('.', ',')
    ax.text(0.97, 0.93, textstr, transform=ax.transAxes, fontsize=8.5,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#555555', alpha=0.9))
            
    ax.set_xticks(range(len(speakers)))
    ax.set_xticklabels(signer_labels)
    if p['ax'] in axes[1,:]:
        ax.set_xlabel('ID Penutur', fontsize=10)
    ax.set_ylabel('Nilai koordinat (ternormalisasi)', fontsize=10)
    ax.set_title(p['title'], fontsize=11, fontweight='bold', pad=12)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', fontsize=9, edgecolor='#555555', framealpha=1)

fig.suptitle('Distribusi Posisi Pergelangan Tangan per Penutur — Seluruh Dataset', 
             fontsize=15, fontweight='bold', y=1.03)
fig.text(0.5, 0.985, 'Centroid per video; garis tebal = median penutur; garis putus-putus = median dataset',
         ha='center', fontsize=11, style='italic')

plt.tight_layout()
save_path_1 = os.path.join(OUTPUT_DIR, 'spatial_distribution_boxplot.png')
plt.savefig(save_path_1, bbox_inches='tight', dpi=300)
plt.show()
print(f"[SAVED] {save_path_1}")

# ── 3. Gambar 2: Heatmap Jarak Pairwise & Uji Signifikansi ──────────────────
pairs = list(itertools.combinations(speakers, 2))
results_lh = []
results_rh = []

df_lh_x = df.pivot(index='sentence', columns='signer', values='lh_x').dropna()
df_lh_y = df.pivot(index='sentence', columns='signer', values='lh_y').dropna()
df_rh_x = df.pivot(index='sentence', columns='signer', values='rh_x').dropna()
df_rh_y = df.pivot(index='sentence', columns='signer', values='rh_y').dropna()

for u, v in pairs:
    # Tangan Kiri
    dist_l = np.sqrt((df_lh_x[u] - df_lh_x[v])**2 + (df_lh_y[u] - df_lh_y[v])**2)
    mean_dist_l = dist_l.mean()
    
    stat_lx, p_lx = wilcoxon(df_lh_x[u], df_lh_x[v])
    stat_ly, p_ly = wilcoxon(df_lh_y[u], df_lh_y[v])
    p_val_l = min(p_lx, p_ly) 
    results_lh.append({'u': u, 'v': v, 'dist': mean_dist_l, 'p': p_val_l})
    
    # Tangan Kanan
    dist_r = np.sqrt((df_rh_x[u] - df_rh_x[v])**2 + (df_rh_y[u] - df_rh_y[v])**2)
    mean_dist_r = dist_r.mean()
    
    stat_rx, p_rx = wilcoxon(df_rh_x[u], df_rh_x[v])
    stat_ry, p_ry = wilcoxon(df_rh_y[u], df_rh_y[v])
    p_val_r = min(p_rx, p_ry)
    results_rh.append({'u': u, 'v': v, 'dist': mean_dist_r, 'p': p_val_r})

res_lh_df = pd.DataFrame(results_lh)
res_rh_df = pd.DataFrame(results_rh)

_, p_fdr_lh, _, _ = smm.multipletests(res_lh_df['p'], method='fdr_bh')
_, p_fdr_rh, _, _ = smm.multipletests(res_rh_df['p'], method='fdr_bh')
res_lh_df['p_fdr'] = p_fdr_lh
res_rh_df['p_fdr'] = p_fdr_rh

def get_stars(p):
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return ''

dist_mat_lh = np.full((len(speakers), len(speakers)), np.nan)
annot_mat_lh = np.full((len(speakers), len(speakers)), "", dtype=object)
dist_mat_rh = np.full((len(speakers), len(speakers)), np.nan)
annot_mat_rh = np.full((len(speakers), len(speakers)), "", dtype=object)

for _, row in res_lh_df.iterrows():
    ui = speakers.index(row['u'])
    vi = speakers.index(row['v'])
    stars = get_stars(row['p_fdr'])
    dist_mat_lh[ui, vi] = row['dist']
    annot_mat_lh[ui, vi] = f"{row['dist']:.3f} {stars}"

for _, row in res_rh_df.iterrows():
    ui = speakers.index(row['u'])
    vi = speakers.index(row['v'])
    stars = get_stars(row['p_fdr'])
    dist_mat_rh[ui, vi] = row['dist']
    annot_mat_rh[ui, vi] = f"{row['dist']:.3f} {stars}"

mean_lh_dist = res_lh_df['dist'].mean()
std_lh_dist = res_lh_df['dist'].std()
mean_rh_dist = res_rh_df['dist'].mean()
std_rh_dist = res_rh_df['dist'].std()

fig = plt.figure(figsize=(13, 6))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.25)
ax_l = plt.subplot(gs[0])
ax_r = plt.subplot(gs[1])

for i in range(len(speakers)):
    for j in range(len(speakers)):
        if i >= j: 
            if i == j:
                annot_mat_lh[i, j] = "—"
                annot_mat_rh[i, j] = "—"

def plot_heatmap_mpl(ax, data, annot, cmap, title, mean_dist, std_dist, is_left=True):
    # Setup custom mesh
    masked_data = np.ma.masked_invalid(data)
    im = ax.imshow(masked_data, cmap=cmap, vmin=0, vmax=0.25, aspect='auto')
    
    # Grid lines
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    
    # Text
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = annot[i, j]
            if pd.isna(data[i,j]) and val == "":
                continue
            color = 'white' if not pd.isna(data[i,j]) and data[i,j] > 0.15 else 'black'
            if val == "—":
                color = 'black'
            ax.text(j, i, val, ha="center", va="center", color=color, fontsize=10)
            
    ax.set_xticks(np.arange(len(speakers)))
    ax.set_xticklabels(signer_labels)
    ax.set_yticks(np.arange(len(speakers)))
    ax.set_yticklabels(signer_labels)
    
    ax.set_facecolor('#F8F9FA') 
    ax.tick_params(axis='both', which='major', length=0)
    ax.set_ylabel('Penutur', fontsize=11)
    ax.set_xlabel('Penutur', fontsize=11)
    
    # Title Tag
    c_tag = '#8B0000' if is_left else '#00008B'
    c_txt = '(a) Tangan Kiri' if is_left else '(b) Tangan Kanan'
    ax.text(0, -0.22, f" {c_txt} ", fontsize=13, fontweight='bold', color=c_tag, 
            bbox=dict(facecolor=c_tag, alpha=0.15, edgecolor='none', boxstyle='square,pad=0.4'))
    ax.text(1, -0.2, f"Rata-rata jarak antar penutur = {mean_dist:.4f} \u00B1 {std_dist:.4f}".replace('.', ','), 
            fontsize=9.5, color=c_tag, transform=ax.transAxes, ha='right',
            bbox=dict(facecolor='white', edgecolor=c_tag, boxstyle='round,pad=0.4'))
            
    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Mean Euclidean\nDistance\n(ternormalisasi)', rotation=-90, va="bottom", fontsize=10)
    cbar.outline.set_visible(False)

plot_heatmap_mpl(ax_l, dist_mat_lh, annot_mat_lh, 'YlOrRd', '(a) Tangan Kiri', mean_lh_dist, std_lh_dist, True)
plot_heatmap_mpl(ax_r, dist_mat_rh, annot_mat_rh, 'Blues', '(b) Tangan Kanan', mean_rh_dist, std_rh_dist, False)

for ax in [ax_l, ax_r]:
    for spine in ax.spines.values():
        spine.set_visible(False)

fig.suptitle('Perbedaan Posisi Pergelangan Tangan antar Penutur', 
             fontsize=16, fontweight='bold', y=1.05)
fig.text(0.5, 0.98, 'Rata-rata jarak pairwise pada seluruh kalimat (nilai lebih besar = perbedaan posisi lebih besar)',
         ha='center', fontsize=11, style='italic')

fig.text(0.5, -0.02, '* Signifikansi dihitung menggunakan uji Wilcoxon signed-rank dengan koreksi FDR.\\n*** p < 0.001, ** p < 0.01, * p < 0.05',
         ha='center', fontsize=10, style='italic')

plt.tight_layout()
save_path_2 = os.path.join(OUTPUT_DIR, 'spatial_pairwise_heatmap.png')
plt.savefig(save_path_2, bbox_inches='tight', dpi=300)
plt.show()
print(f"[SAVED] {save_path_2}")
