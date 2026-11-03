"""
Publication-Quality System Architecture Diagram for IEEE NLP Paper
Refined coordinates: Perfect margins, zero label clipping, elegant academic typesetting.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

# Configure academic serif typography
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'figure.dpi': 300,
})

# Canvas: width 9.6 in, height 5.8 in
fig, ax = plt.subplots(figsize=(9.6, 5.8))
ax.set_xlim(0, 96)
ax.set_ylim(0, 58)
ax.axis('off')
fig.patch.set_facecolor('#ffffff')
ax.set_facecolor('#ffffff')

# Academic Color Palette
C_DARK    = '#1A252C'
C_MUTED   = '#5A6B7C'
C_BORDER  = '#34495E'

def draw_academic_block(x, y, w, h, title, subtitle=None, bg_color='#F8F9FA', border_color=C_BORDER, lw=1.1, rounding=0.4):
    box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.1,rounding_size={rounding}",
                          facecolor=bg_color, edgecolor=border_color, linewidth=lw, zorder=2)
    ax.add_patch(box)
    
    if subtitle:
        ax.text(x + w/2, y + h/2 + 1.15, title, ha='center', va='center',
                fontsize=8.8, fontweight='bold', color=C_DARK, zorder=4)
        ax.text(x + w/2, y + h/2 - 1.15, subtitle, ha='center', va='center',
                fontsize=7.4, fontstyle='italic', color=C_MUTED, zorder=4)
    else:
        ax.text(x + w/2, y + h/2, title, ha='center', va='center',
                fontsize=8.8, fontweight='bold', color=C_DARK, zorder=4)

def draw_flow_arrow(x1, y1, x2, y2, label=None, label_y_offset=0.8, lw=1.1, color='#2C3E50'):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle='-|>,head_length=4.2,head_width=3.0',
                            connectionstyle='arc3,rad=0',
                            color=color, linewidth=lw, zorder=5)
    ax.add_patch(arrow)
    if label:
        ax.text((x1 + x2)/2, (y1 + y2)/2 + label_y_offset, label, fontsize=6.8, fontstyle='italic',
                color=C_MUTED, va='center', ha='center', zorder=6)

def draw_dashed_stage(x, y, w, h, stage_num, title):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.5",
                          facecolor='none', edgecolor='#95A5A6', linewidth=0.7,
                          linestyle='--', zorder=1)
    ax.add_patch(rect)
    header_text = f"Stage {stage_num}: {title}"
    ax.text(x + 1.5, y + h - 0.9, header_text, fontsize=7.8, fontweight='bold',
            color='#34495E', va='top', ha='left', zorder=3,
            bbox=dict(boxstyle='square,pad=0.15', facecolor='#ffffff', edgecolor='none'))

# ==============================================================================
# TIER 1 (TOP): INPUT & MULTI-FEATURE ENCODING (y: 29.5 to 56.5)
# ==============================================================================

# --- Stage 1: Linguistic Input & Token Representation (Left: x=1.5 to 44) ---
draw_dashed_stage(1.5, 29.5, 43.5, 27.0, "1", "Linguistic Input & Feature Alignment")

# Input Sentence
draw_academic_block(3.2, 50.2, 40.0, 4.6,
                    'Input Text: "LockBit breached server via CVE-2023-38606..."',
                    'Raw Cybersecurity Incident Report',
                    bg_color='#FFFFFF', border_color='#7F8C8D', lw=1.0)

# Token Table Box
table_y = 31.0
table_h = 15.0
table_w = 40.0
table_x = 3.2

t_box = FancyBboxPatch((table_x, table_y), table_w, table_h, boxstyle="round,pad=0.08,rounding_size=0.3",
                       facecolor='#FAFAFA', edgecolor='#BDC3C7', linewidth=0.85, zorder=2)
ax.add_patch(t_box)

ax.text(table_x + 1.8, table_y + table_h - 1.5, "Linguistic Feature Extraction (spaCy Pipeline):",
        fontsize=7.4, fontweight='bold', color='#2C3E50', zorder=4)

# Tokens and Features alignment
col_x = [11.0, 18.0, 25.0, 31.8, 38.6]
col_tokens = ['"LockBit"', '"breached"', '"server"', '"via"', '"CVE-..."']
col_pos    = ['PROPN', 'VERB', 'NOUN', 'ADP', 'PROPN']
col_ner    = ['ORG', 'O', 'O', 'O', 'O']

# Row Labels inside the table
ax.text(4.2, table_y + 10.0, "Word:", fontsize=7.0, fontweight='bold', color=C_MUTED, va='center', ha='left', zorder=4)
ax.text(4.2, table_y + 6.4, "POS (20):", fontsize=7.0, fontweight='bold', color=C_MUTED, va='center', ha='left', zorder=4)
ax.text(4.2, table_y + 2.8, "NER (4):", fontsize=7.0, fontweight='bold', color=C_MUTED, va='center', ha='left', zorder=4)

for i in range(len(col_tokens)):
    ax.text(col_x[i], table_y + 10.0, col_tokens[i], fontsize=7.3, fontweight='bold', color='#1A252C', va='center', ha='center', zorder=4)
    ax.text(col_x[i], table_y + 6.4, col_pos[i], fontsize=7.0, color='#27AE60', fontweight='bold', va='center', ha='center', zorder=4)
    ax.text(col_x[i], table_y + 2.8, col_ner[i], fontsize=7.0, color='#2980B9', fontweight='bold', va='center', ha='center', zorder=4)

draw_flow_arrow(23.2, 50.2, 23.2, 46.0)


# --- Stage 2: Multi-Feature Transformer Backbone (Right: x=46.5 to 94.5) ---
draw_dashed_stage(46.5, 29.5, 48.0, 27.0, "2", "Multi-Feature Embedding & Fusion")

# 1. Transformer Encoder Backbone
draw_academic_block(48.5, 48.6, 44.0, 6.2,
                    r"Transformer Backbone (SecBERT / BERT / RoBERTa)",
                    r"Subword Contextual Embeddings:  $H_{\mathrm{trans}} \in \mathbb{R}^{B \times L \times 768}$",
                    bg_color='#EDF4FA', border_color='#2980B9', lw=1.1)

# 2. POS & NER Embedding Layers
draw_academic_block(48.5, 39.8, 21.0, 5.8,
                    r"POS Embedding",
                    r"$E_{\mathrm{pos}} \in \mathbb{R}^{B \times L \times 32}$",
                    bg_color='#F3F6F4', border_color='#27AE60', lw=1.0)

draw_academic_block(71.5, 39.8, 21.0, 5.8,
                    r"General NER Embedding",
                    r"$E_{\mathrm{ner}} \in \mathbb{R}^{B \times L \times 32}$",
                    bg_color='#F3F6F4', border_color='#16A085', lw=1.0)

# Connect Stage 1 to Stage 2
draw_flow_arrow(43.2, 42.0, 48.5, 51.5) # Words to Transformer
draw_flow_arrow(43.2, 37.4, 48.5, 42.7) # POS to POS Emb
draw_flow_arrow(43.2, 33.8, 71.5, 42.7) # NER to NER Emb

# 3. Concatenation Fusion Layer
draw_academic_block(48.5, 31.2, 44.0, 5.6,
                    r"Feature Concatenation:  $H_{\mathrm{combined}} = [\, H_{\mathrm{trans}} \;;\; E_{\mathrm{pos}} \;;\; E_{\mathrm{ner}} \,]$",
                    r"Dense Joint Representation Vector  $\in \mathbb{R}^{B \times L \times 832}$",
                    bg_color='#F2F4F4', border_color='#34495E', lw=1.1)

draw_flow_arrow(70.5, 48.6, 70.5, 45.6)
draw_flow_arrow(59.0, 39.8, 59.0, 36.8)
draw_flow_arrow(82.0, 39.8, 82.0, 36.8)


# ==============================================================================
# PIPELINE CONNECTOR (Between Tier 1 and Tier 2)
# ==============================================================================
# Clean orthogonal path through the channel at y=27.5 (between y=25.5 and y=29.5)
ax.plot([70.5, 70.5, 23.2, 23.2], [31.2, 27.5, 27.5, 22.8], color='#34495E', lw=1.2, zorder=5)
arrow_down_head = FancyArrowPatch((23.2, 23.2), (23.2, 22.3),
                                  arrowstyle='-|>,head_length=4.5,head_width=3.2',
                                  color='#34495E', linewidth=1.2, zorder=6)
ax.add_patch(arrow_down_head)

ax.text(47.0, 27.5, r"Joint Representation  $H_{\mathrm{combined}} \in \mathbb{R}^{832}$",
        fontsize=7.5, fontweight='bold', fontstyle='italic', color='#2C3E50',
        ha='center', va='center', zorder=8,
        bbox=dict(boxstyle='round,pad=0.22', facecolor='#FFFFFF', edgecolor='#BDC3C7', lw=0.7))


# ==============================================================================
# TIER 2 (BOTTOM): CLASSIFICATION, ENSEMBLE & VITERBI DECODING (y: 1.5 to 25.5)
# ==============================================================================

# --- Stage 3: Classification & Cost-Sensitive Loss (Left: x=1.5 to 44) ---
draw_dashed_stage(1.5, 1.5, 43.5, 24.0, "3", "Classification & Cost-Sensitive Learning")

# Linear Classification Head
draw_academic_block(3.2, 14.5, 40.0, 7.8,
                    r"Linear Classification Head",
                    r"Dropout ($p=0.1$) $\rightarrow$ Linear $\rightarrow$ Emission Logits $Z \in \mathbb{R}^{B \times L \times 63}$",
                    bg_color='#FFFDF5', border_color='#D4AC0D', lw=1.1)

# Training Cost-Sensitive Loss Box
draw_academic_block(3.2, 3.2, 40.0, 7.8,
                    r"Training: Cost-Sensitive Cross-Entropy Loss",
                    r"Downweights 'O' background: $w_O = 0.35$,  Targets: $w_c = 1.0$" + "\n" + r"Compels optimization gradient to focus on scarce arguments",
                    bg_color='#FDEDEC', border_color='#E74C3C', lw=1.0)

draw_flow_arrow(23.2, 14.5, 23.2, 11.0, label="Backpropagation (Training)", label_y_offset=0.0)


# --- Stage 4: Inference, Ensemble & Viterbi Decoding (Right: x=46.5 to 94.5) ---
draw_dashed_stage(46.5, 1.5, 48.0, 24.0, "4", "Word-Level Ensemble & Constrained Decoding")

# Multi-Model Softmax Ensembling
draw_academic_block(48.5, 16.5, 44.0, 6.5,
                    r"Word-Level Softmax Ensemble Across Backbones",
                    r"$P_{\mathrm{ens}}(w) = \frac{1}{|M|} \sum_{m \in M} P_m(w), \quad M = \{\mathrm{SecBERT}, \mathrm{BERT}, \mathrm{RoBERTa}\}$",
                    bg_color='#F4F2F9', border_color='#8E44AD', lw=1.1)

# Arrow from Classification Head to Ensemble (Softmax probability)
draw_flow_arrow(43.2, 18.5, 48.5, 18.5, label=r"Softmax $P_m(w)$", label_y_offset=1.2)

# Viterbi Constrained Decoder
draw_academic_block(48.5, 9.8, 44.0, 5.0,
                    r"Viterbi Constrained BIO Decoder",
                    r"Enforces $63 \times 63$ Boolean Transition Matrix $T$; Illegal transitions $\rightarrow -\infty$",
                    bg_color='#FCF3CF', border_color='#F39C12', lw=1.1)

draw_flow_arrow(70.5, 16.5, 70.5, 14.8)

# Final Extracted Arguments Output Box
draw_academic_block(48.5, 3.0, 44.0, 5.2,
                    r"Extracted Event Arguments & Role Spans",
                    r"[Attacker] LockBit   |   [Victim] server   |   [CVE] CVE-2023-38606",
                    bg_color='#EAF7EE', border_color='#27AE60', lw=1.1)

draw_flow_arrow(70.5, 9.8, 70.5, 8.2)

plt.tight_layout(pad=0.15)

out_fig = Path(r"d:\Kasetsart\NLP\LPLLL\figures\system_architecture.png")
plt.savefig(out_fig, dpi=300, bbox_inches='tight', pad_inches=0.03)
print(f"Refined academic figure saved at: {out_fig}")
