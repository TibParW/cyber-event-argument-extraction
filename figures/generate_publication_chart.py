"""
Publication-Quality F1 Comparison Chart for IEEE Paper (Fig. 2)
Logically sorted: Base -> +POS -> +NER -> +Both (POS+NER) -> Ensemble.
Clean Times New Roman academic typography.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Configure academic serif typography
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'figure.dpi': 300,
})

csv_path = Path(r"d:\Kasetsart\NLP\LPLLL\master_ablation_results_v3.csv")
df = pd.read_csv(csv_path)

# Custom sort order: for each backbone, group: NONE (Base) -> POS -> NER -> POS_NER (Both)
# And Ensemble at the end
feat_order = {'NONE': 0, 'POS': 1, 'NER': 2, 'POS_NER': 3, 'COMBINED': 4}
model_order = {'SecBERT': 0, 'bert-base-uncased': 1, 'roberta-base': 2, 'ENSEMBLE (Multi-Backbone)': 3}

df['m_rank'] = df['Model'].map(model_order)
df['f_rank'] = df['Features'].map(feat_order)
df_sorted = df.sort_values(by=['m_rank', 'f_rank']).reset_index(drop=True)

# Canvas: width 8.2 in, height 4.2 in
fig, ax = plt.subplots(figsize=(8.2, 4.2))

# Academic Colors
C_STRICT  = '#1B365D'   # Deep Navy Blue
C_RELAXED = '#B83A2E'   # Muted Academic Crimson
C_DARK    = '#1A252C'
C_MUTED   = '#5A6B7C'

n_models = len(df_sorted)
x = np.arange(n_models)
bar_w = 0.36

# Clean feature labels
feat_labels = []
for _, r in df_sorted.iterrows():
    f = str(r['Features']).strip()
    if f == 'NONE':
        feat_labels.append('Base')
    elif f == 'POS':
        feat_labels.append('+POS')
    elif f == 'NER':
        feat_labels.append('+NER')
    elif f == 'POS_NER':
        feat_labels.append('+Both')
    elif f == 'COMBINED':
        feat_labels.append('All')
    else:
        feat_labels.append(f)

# Plot bars
rects1 = ax.bar(x - bar_w/2, df_sorted['F1 (%)'], bar_w,
                label='Strict F1 (Exact Match)', color=C_STRICT,
                edgecolor='#0E1E33', linewidth=0.9, zorder=3)
rects2 = ax.bar(x + bar_w/2, df_sorted['Relaxed F1 (%)'], bar_w,
                label='Relaxed F1 (NIST TAC Overlap)', color=C_RELAXED,
                edgecolor='#681E16', linewidth=0.9, zorder=3)

# Highlight Ensemble
ax.axvspan(11.5, 12.5, facecolor='#F4F2F9', edgecolor='#8E44AD',
          linestyle='--', linewidth=1.1, alpha=0.75, zorder=1)

# Annotate Ensemble peak
f1_ens_strict = df_sorted['F1 (%)'].iloc[-1]
f1_ens_relaxed = df_sorted['Relaxed F1 (%)'].iloc[-1]
ax.text(12 - bar_w/2, f1_ens_strict + 1.2, f'{f1_ens_strict:.1f}%',
        ha='center', va='bottom', fontsize=9.2, fontweight='bold', color=C_STRICT)
ax.text(12 + bar_w/2, f1_ens_relaxed + 1.2, f'{f1_ens_relaxed:.1f}%',
        ha='center', va='bottom', fontsize=9.2, fontweight='bold', color=C_RELAXED)

# Annotate best single-model (RoBERTa + POS: row 9 in sorted df)
best_single_idx = df_sorted[df_sorted['Model'] != 'ENSEMBLE (Multi-Backbone)']['F1 (%)'].idxmax()
best_single_f1 = df_sorted['F1 (%)'].iloc[best_single_idx]
ax.text(best_single_idx - bar_w/2, best_single_f1 + 1.2, f'{best_single_f1:.1f}%',
        ha='center', va='bottom', fontsize=8.0, fontweight='bold', color=C_MUTED)

# Y-Axis
ax.set_ylabel('F1 Score (%)', fontsize=12.0, fontweight='bold', color=C_DARK)
ax.set_ylim(0, 85)
ax.set_yticks([0, 20, 40, 60, 80])
ax.tick_params(axis='y', labelsize=10.0, colors=C_DARK)

# X-Axis (Feature ticks)
ax.set_xticks(x)
ax.set_xticklabels(feat_labels, fontsize=9.2, fontweight='bold', color=C_DARK)
ax.tick_params(axis='x', length=3.5, pad=3)

# Backbone Group Brackets
y_bracket = -11.0
ax.plot([-0.3, 3.3], [y_bracket, y_bracket], color='#34495E', lw=1.3, clip_on=False)
ax.text(1.5, y_bracket - 3.2, 'SecBERT', ha='center', va='top', fontsize=10.5, fontweight='bold', color=C_DARK)

ax.plot([3.7, 7.3], [y_bracket, y_bracket], color='#34495E', lw=1.3, clip_on=False)
ax.text(5.5, y_bracket - 3.2, 'BERT-base', ha='center', va='top', fontsize=10.5, fontweight='bold', color=C_DARK)

ax.plot([7.7, 11.3], [y_bracket, y_bracket], color='#34495E', lw=1.3, clip_on=False)
ax.text(9.5, y_bracket - 3.2, 'RoBERTa-base', ha='center', va='top', fontsize=10.5, fontweight='bold', color=C_DARK)

ax.plot([11.7, 12.3], [y_bracket, y_bracket], color='#8E44AD', lw=1.6, clip_on=False)
ax.text(12.0, y_bracket - 3.2, 'Ensemble', ha='center', va='top', fontsize=10.5, fontweight='bold', color='#6C3483')

# Clean academic spines & grid
ax.grid(axis='y', linestyle=':', color='#BDC3C7', alpha=0.6, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#7F8C8D')
ax.spines['bottom'].set_color('#7F8C8D')

# Legend
leg = ax.legend(loc='upper left', frameon=True, fontsize=9.5, framealpha=0.92, edgecolor='#BDC3C7')
leg.get_frame().set_linewidth(0.8)

plt.subplots_adjust(bottom=0.22, top=0.95, left=0.08, right=0.97)

out_chart = Path(r"d:\Kasetsart\NLP\LPLLL\figures\ablation_f1_comparison_chart.png")
plt.savefig(out_chart, dpi=300)
print(f"Publication-ready comparison chart saved to: {out_chart}")
