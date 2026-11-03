"""
Clean, publication-quality system architecture figure
Style: real NLP conference papers (ACL / EMNLP / AAAI / IEEE)

Principles:
  - Minimal color (white + very light gray + one subtle blue accent)
  - Serif font (Times New Roman) matching IEEE body text
  - Thin black borders, straight arrows, no fancy decorations
  - Vertical top-to-bottom flow, fits IEEE single-column (3.3 in)
  - 300 DPI, tight layout
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'figure.dpi': 300,
})

fig, ax = plt.subplots(figsize=(6.6, 9.2))
ax.set_xlim(-3, 68)
ax.set_ylim(0, 92)
ax.axis('off')
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# ── Colors ──
WHITE  = '#FFFFFF'
LGRAY  = '#F4F4F4'
MGRAY  = '#E6E6E6'
BLACK  = '#1A1A1A'
DGRAY  = '#505050'
ARROW  = '#333333'
BLUE   = '#DAE8F5'  # subtle accent for key components


def box(x, y, w, h, label, sub=None, fill=WHITE, edge=BLACK,
        lw=1.0, fs=9.5, sfs=7.5, bold=True, rd=0.3):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.12,rounding_size={rd}",
        facecolor=fill, edgecolor=edge, linewidth=lw, zorder=3)
    ax.add_patch(patch)
    if sub:
        ax.text(x + w/2, y + h/2 + 1.3, label,
                ha='center', va='center', fontsize=fs,
                fontweight='bold' if bold else 'normal', color=BLACK, zorder=4)
        ax.text(x + w/2, y + h/2 - 1.2, sub,
                ha='center', va='center', fontsize=sfs,
                fontstyle='italic', color=DGRAY, zorder=4)
    else:
        ax.text(x + w/2, y + h/2, label,
                ha='center', va='center', fontsize=fs,
                fontweight='bold' if bold else 'normal', color=BLACK, zorder=4)


def arrow_d(x, y1, y2):
    ax.annotate('', xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle='-|>', lw=1.0, color=ARROW,
                                mutation_scale=9, shrinkA=1.5, shrinkB=1.5),
                zorder=5)


def dashed_group(x, y, w, h, label):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.15,rounding_size=0.4",
        facecolor='none', edgecolor='#999999',
        linewidth=0.6, linestyle=(0, (4, 3)), zorder=1)
    ax.add_patch(rect)
    # Place label inside top-left, shifted right to avoid clipping
    ax.text(x + 2, y + h - 1.2, label,
            fontsize=7.5, fontweight='bold', color='#777777',
            va='top', ha='left', zorder=6,
            bbox=dict(boxstyle='square,pad=0.15', facecolor='white',
                      edgecolor='none', alpha=0.85))


# ── Layout constants ──
CX   = 33     # horizontal center
BW   = 18     # standard box width
BWS  = 16     # smaller box width
SP   = 1.5    # vertical spacing between boxes

# Three columns for parallel features
C1 = 3        # left col start
C2 = 24       # middle col start
C3 = 45       # right col start
CW = 18       # column box width


# ======================================================================
# STAGE 1: Input & Feature Extraction
# ======================================================================
S1_TOP = 74
S1_H   = 17
dashed_group(1, S1_TOP, 64, S1_H, 'Stage 1:  Input & Feature Extraction')

# Input
Y_IN = S1_TOP + S1_H - 4.5
box(CX - BW/2, Y_IN, BW, 3.5,
    'Cybersecurity News Article',
    fill=LGRAY, fs=8.5, bold=False, rd=0.15)

# 3 parallel extractors
Y_FE = Y_IN - 8
box(C1, Y_FE, CW, 5.5,
    'Subword Tokenizer', 'WordPiece / BPE',
    fill=WHITE, sfs=7)
box(C2, Y_FE, CW, 5.5,
    'POS Tagger', 'spaCy  (20 tags)',
    fill=WHITE, sfs=7)
box(C3, Y_FE, CW, 5.5,
    'General NER', 'spaCy  (4 classes)',
    fill=WHITE, sfs=7)

# Arrows: input -> 3 extractors
arrow_d(C1 + CW/2, Y_IN, Y_FE + 5.5)
arrow_d(C2 + CW/2, Y_IN, Y_FE + 5.5)
arrow_d(C3 + CW/2, Y_IN, Y_FE + 5.5)


# ======================================================================
# STAGE 2: Multi-Feature Encoding
# ======================================================================
S2_TOP = 53
S2_H   = 19.5
dashed_group(1, S2_TOP, 64, S2_H, 'Stage 2:  Multi-Feature Encoding')

# 3 parallel encoders
Y_EN = S2_TOP + S2_H - 5
box(C1, Y_EN, CW, 5.5,
    'Transformer Encoder', 'SecBERT / BERT / RoBERTa',
    fill=BLUE, sfs=7)
box(C2, Y_EN, CW, 5.5,
    'POS Embedding', 'Learnable  (dim = 32)',
    fill=WHITE, sfs=7)
box(C3, Y_EN, CW, 5.5,
    'NER Embedding', 'Learnable  (dim = 32)',
    fill=WHITE, sfs=7)

# Dimension labels (below each encoder box)
ax.text(C1 + CW/2, Y_EN - 0.6, '768-d',
        fontsize=6.5, color=DGRAY, ha='center', va='top', fontstyle='italic')
ax.text(C2 + CW/2, Y_EN - 0.6, '32-d',
        fontsize=6.5, color=DGRAY, ha='center', va='top', fontstyle='italic')
ax.text(C3 + CW/2, Y_EN - 0.6, '32-d',
        fontsize=6.5, color=DGRAY, ha='center', va='top', fontstyle='italic')

# Arrows: Stage 1 -> Stage 2
arrow_d(C1 + CW/2, Y_FE, Y_EN + 5.5)
arrow_d(C2 + CW/2, Y_FE, Y_EN + 5.5)
arrow_d(C3 + CW/2, Y_FE, Y_EN + 5.5)

# Concatenation bar
Y_CAT = S2_TOP + 1.5
box(C1, Y_CAT, 60, 3.5,
    'Concatenation:   H = [ H_trans ; E_pos ; E_ner ]     (832-d)',
    fill=MGRAY, fs=8, bold=False, rd=0.12)

# Arrows: 3 encoders -> concat
arrow_d(C1 + CW/2, Y_EN, Y_CAT + 3.5)
arrow_d(C2 + CW/2, Y_EN, Y_CAT + 3.5)
arrow_d(C3 + CW/2, Y_EN, Y_CAT + 3.5)


# ======================================================================
# STAGE 3: Classification
# ======================================================================
S3_TOP = 39
S3_H   = 12.5
dashed_group(1, S3_TOP, 64, S3_H, 'Stage 3:  Classification')

Y_CLS = S3_TOP + S3_H - 5.5
box(C1, Y_CLS, 60, 5.5,
    'Classification Head',
    'Dropout (p = 0.1)  +  Linear Projection  \u2192  63 logits',
    fill=WHITE, sfs=7.5)

arrow_d(CX, Y_CAT, Y_CLS + 5.5)

# Loss
Y_LOSS = S3_TOP + 1
box(C1, Y_LOSS, 60, 3.5,
    'Training Loss:   Weighted Cross-Entropy   (w_O = 0.35,  w_c = 1.0)',
    fill=WHITE, fs=7.5, bold=False, rd=0.12, lw=0.7)

arrow_d(CX, Y_CLS, Y_LOSS + 3.5)


# ======================================================================
# STAGE 4: Ensemble & Decoding
# ======================================================================
S4_TOP = 2
S4_H   = 35.5
dashed_group(1, S4_TOP, 64, S4_H, 'Stage 4:  Ensemble & Decoding')

# "x3 backbones" annotation
Y_X3 = S4_TOP + S4_H - 4
ax.text(CX, Y_X3, '\u00d7 3 backbones  (Stages 1\u20133 repeated per model)',
        fontsize=7.5, ha='center', va='center', color=DGRAY,
        fontstyle='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor=LGRAY,
                  edgecolor='#BBBBBB', lw=0.5),
        zorder=5)

# Dotted separator
ax.plot([4, 62], [Y_X3 - 2, Y_X3 - 2],
        linestyle=':', color='#AAAAAA', lw=0.7, zorder=2)

arrow_d(CX, Y_LOSS, Y_X3 + 1.5)

# Ensemble
Y_ENS = Y_X3 - 9
box(C1, Y_ENS, 60, 5.5,
    'Word-Level Softmax Ensemble',
    'P_ens(w) = (1/|M|) \u2211 P_m(w)     M = {SecBERT, BERT, RoBERTa}',
    fill=BLUE, sfs=7.5)

arrow_d(CX, Y_X3 - 2, Y_ENS + 5.5)

# Viterbi
Y_VIT = Y_ENS - 7.5
box(C1, Y_VIT, 60, 5.5,
    'Viterbi Constrained BIO Decoder',
    '63 \u00d7 63 transition matrix;   illegal transitions \u2192 \u2013\u221e',
    fill=WHITE, sfs=7.5)

arrow_d(CX, Y_ENS, Y_VIT + 5.5)

# Output
Y_OUT = Y_VIT - 5.5
box(C1, Y_OUT, 60, 3.8,
    'Extracted Event Arguments',
    'Attacker  |  Victim  |  Tool  |  CVE  |  Time  |  Place  |  ...',
    fill=LGRAY, fs=9, sfs=7.5, rd=0.15)

arrow_d(CX, Y_VIT, Y_OUT + 3.8)


# ── Save ──
plt.tight_layout(pad=0.2)
out = Path(r"d:\Kasetsart\NLP\LPLLL\figures\system_architecture.png")
plt.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.06,
            facecolor='white', edgecolor='none')
print(f"Saved: {out}  ({out.stat().st_size / 1024:.0f} KB)")
