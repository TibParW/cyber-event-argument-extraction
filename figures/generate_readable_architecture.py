import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

# Create figure optimized for IEEE single column:
# Canvas width 7.4 inches, height 6.6 inches (aspect ratio ~1.12)
fig, ax = plt.subplots(figsize=(7.4, 6.6), dpi=300)
ax.set_xlim(0, 7.4)
ax.set_ylim(0, 6.6)
ax.axis('off')

fig.patch.set_facecolor('#ffffff')
ax.set_facecolor('#ffffff')

# Helper functions with large, crisp, legible typography
def draw_card(ax, x, y, w, h, title, bg_color='#f8fafc', border_color='#94a3b8', title_color='#0f172a'):
    """Draws outer stage card"""
    card = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.15",
                          facecolor=bg_color, edgecolor=border_color, linewidth=1.6, zorder=1)
    ax.add_patch(card)
    ax.text(x + w/2, y + h - 0.24, title, ha='center', va='center',
            fontsize=12.5, fontweight='bold', color=title_color, family='sans-serif', zorder=2)

def draw_box(ax, x, y, w, h, title, lines, bg_color, border_color, title_color='#0f172a', badge=None, badge_bg='#e2e8f0'):
    """Draws inner module box with large, high-contrast text"""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.1",
                          facecolor=bg_color, edgecolor=border_color, linewidth=1.4, zorder=3)
    ax.add_patch(box)
    
    t_y = y + h - 0.20 if lines else y + h/2
    ax.text(x + w/2, t_y, title, ha='center', va='center',
            fontsize=11.0, fontweight='bold', color=title_color, family='sans-serif', zorder=4)
    
    if lines:
        for i, line in enumerate(lines):
            l_y = t_y - 0.23 - (i * 0.20)
            ax.text(x + w/2, l_y, line, ha='center', va='center',
                    fontsize=9.2, color='#1e293b', family='sans-serif', zorder=4)
            
    if badge:
        b_w, b_h = 1.05, 0.24
        b_x, b_y = x + w - b_w - 0.05, y + 0.05
        badge_box = FancyBboxPatch((b_x, b_y), b_w, b_h, boxstyle="round,pad=0.02,rounding_size=0.05",
                                   facecolor=badge_bg, edgecolor=border_color, linewidth=1.0, zorder=5)
        ax.add_patch(badge_box)
        ax.text(b_x + b_w/2, b_y + b_h/2, badge, ha='center', va='center',
                fontsize=8.5, fontweight='bold', color='#0f172a', family='sans-serif', zorder=6)

def draw_arrow(ax, x1, y1, x2, y2, color='#475569', lw=1.8):
    """Draws clear directional arrow"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle='-|>,head_length=5.5,head_width=3.8',
                            connectionstyle='arc3,rad=0',
                            color=color, linewidth=lw, zorder=7)
    ax.add_patch(arrow)

# ==============================================================================
# ROW 1 (TOP): Stage 1 & Stage 2
# ==============================================================================
# Stage 1: Input & Linguistic Feature Extraction (Left)
draw_card(ax, 0.2, 3.45, 3.3, 3.0, "1. INPUT & LINGUISTIC PIPELINE",
          bg_color='#f8fafc', border_color='#64748b', title_color='#0f172a')

draw_box(ax, 0.35, 5.5, 3.0, 0.72, "Raw News Article",
         ['"LockBit breached servers via CVE-2023-..."'],
         bg_color='#ffffff', border_color='#94a3b8')

draw_box(ax, 0.35, 4.45, 3.0, 0.85, "spaCy POS Tagger",
         ["Universal 20 tags (NOUN, VERB, NUM...)"],
         bg_color='#f0fdf4', border_color='#16a34a', title_color='#14532d',
         badge="20 Tags", badge_bg='#dcfce7')

draw_box(ax, 0.35, 3.55, 3.0, 0.75, "spaCy General NER",
         ["4 Coarse Entities (ORG, PER, GPE, LOC)"],
         bg_color='#eff6ff', border_color='#2563eb', title_color='#1e3a8a',
         badge="4 Tags", badge_bg='#dbeafe')

# Arrows inside Stage 1
draw_arrow(ax, 1.85, 5.5, 1.85, 5.3)
draw_arrow(ax, 1.85, 4.45, 1.85, 4.3)

# Stage 2: Multi-Feature Transformer Backbone (Right)
draw_card(ax, 3.9, 3.45, 3.3, 3.0, "2. MULTI-FEATURE ENCODING",
          bg_color='#faf5ff', border_color='#9333ea', title_color='#581c87')

draw_box(ax, 4.05, 5.35, 3.0, 0.88, "Transformer Backbone",
         ["SecBERT / BERT / RoBERTa", "First subword token representation"],
         bg_color='#ffffff', border_color='#10b981', title_color='#065f46',
         badge="Dim: 768", badge_bg='#d1fae5')

draw_box(ax, 4.05, 4.45, 3.0, 0.75, "POS Embedding Layer",
         ["Learnable Lookup Table (20 IDs -> Vector)"],
         bg_color='#f0fdf4', border_color='#16a34a', title_color='#14532d',
         badge="Dim: 32", badge_bg='#dcfce7')

draw_box(ax, 4.05, 3.55, 3.0, 0.75, "NER Embedding Layer",
         ["Learnable Lookup Table (4 IDs -> Vector)"],
         bg_color='#eff6ff', border_color='#2563eb', title_color='#1e3a8a',
         badge="Dim: 32", badge_bg='#dbeafe')

# Arrows from Stage 1 to Stage 2
draw_arrow(ax, 3.35, 5.85, 4.05, 5.85, color='#0284c7', lw=1.8)
draw_arrow(ax, 3.35, 4.82, 4.05, 4.82, color='#16a34a', lw=1.8)
draw_arrow(ax, 3.35, 3.92, 4.05, 3.92, color='#2563eb', lw=1.8)


# ==============================================================================
# PIPELINE CONNECTOR: Curved arrow from Row 1 to Row 2
# ==============================================================================
conn_arrow = FancyArrowPatch((5.55, 3.45), (1.85, 3.15),
                             arrowstyle='-|>,head_length=6,head_width=4.5',
                             connectionstyle='arc3,rad=-0.35',
                             color='#7c3aed', linewidth=2.4, zorder=8)
ax.add_patch(conn_arrow)
ax.text(3.7, 3.25, "Embeddings [768; 32; 32]", ha='center', va='center',
        fontsize=9.5, fontweight='bold', color='#6b21a8', family='sans-serif',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#f3e8ff', edgecolor='#c084fc', lw=1.0), zorder=9)


# ==============================================================================
# ROW 2 (BOTTOM): Stage 3 & Stage 4
# ==============================================================================
# Stage 3: Fusion & Classification Head (Left)
draw_card(ax, 0.2, 0.15, 3.3, 2.9, "3. FUSION & CLASSIFICATION",
          bg_color='#fffbeb', border_color='#d97706', title_color='#78350f')

draw_box(ax, 0.35, 2.05, 3.0, 0.78, "Feature Concatenation",
         ["Concat: [ H_trans ; E_pos ; E_ner ]"],
         bg_color='#f5f3ff', border_color='#8b5cf6', title_color='#4c1d95',
         badge="Dim: 832", badge_bg='#ede9fe')

draw_box(ax, 0.35, 1.15, 3.0, 0.75, "Classification Head",
         ["Dropout (p=0.1) + Linear Projection"],
         bg_color='#ffffff', border_color='#f59e0b', title_color='#b45309',
         badge="63 Logits", badge_bg='#fef3c7')

draw_box(ax, 0.35, 0.25, 3.0, 0.75, "Cost-Sensitive Loss",
         ["Weighted Cross-Entropy (w_c=1.0)"],
         bg_color='#fef2f2', border_color='#ef4444', title_color='#991b1b',
         badge="w_O = 0.35", badge_bg='#fee2e2')

# Arrows inside Stage 3
draw_arrow(ax, 1.85, 2.05, 1.85, 1.90)
draw_arrow(ax, 1.85, 1.15, 1.85, 1.00)

# Stage 4: Decoding & Output Spans (Right)
draw_card(ax, 3.9, 0.15, 3.3, 2.9, "4. DECODING & EXTRACTION",
          bg_color='#f0fdf4', border_color='#16a34a', title_color='#14532d')

draw_box(ax, 4.05, 2.15, 3.0, 0.68, "Multi-Model Ensemble",
         ["Word-Level Softmax: SecBERT + BERT + RoBERTa"],
         bg_color='#ffffff', border_color='#8b5cf6', title_color='#5b21b6')

draw_box(ax, 4.05, 1.35, 3.0, 0.65, "Viterbi Constrained Decoder",
         ["Enforce valid BIO rules (Zero invalid transitions)"],
         bg_color='#fefce8', border_color='#eab308', title_color='#713f12',
         badge="63x63 Matrix", badge_bg='#fef9c3')

draw_box(ax, 4.05, 0.25, 3.0, 0.95, "Extracted Event Spans",
         ["[Event] Ransomware  |  [Attacker] LockBit",
          "[Victim] Servers     |  [CVE] CVE-2023-..."],
         bg_color='#f0fdf4', border_color='#22c55e', title_color='#15803d')

# Arrows between Stage 3 & Stage 4
draw_arrow(ax, 3.35, 1.52, 4.05, 2.45, color='#d97706', lw=1.8) # From logits to ensemble
draw_arrow(ax, 5.55, 2.15, 5.55, 2.00)
draw_arrow(ax, 5.55, 1.35, 5.55, 1.20)

plt.tight_layout()
out_fig = Path(__file__).resolve().parent / "system_architecture.png"
plt.savefig(out_fig, dpi=300, bbox_inches='tight', pad_inches=0.04)
print(f"Saved optimized readable architecture to: {out_fig}")
