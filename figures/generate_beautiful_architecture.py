import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

# Create figure with 300 DPI for high publication quality
fig, ax = plt.subplots(figsize=(14.5, 7.2), dpi=300)
ax.set_xlim(0, 14.5)
ax.set_ylim(0, 7.2)
ax.axis('off')

# Set background to clean white
fig.patch.set_facecolor('#ffffff')
ax.set_facecolor('#ffffff')

def draw_card(ax, x, y, w, h, title, bg_color='#f8fafc', border_color='#cbd5e1', title_color='#0f172a'):
    """Draws a clean modern stage container card"""
    # Outer card
    card = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.15",
                          facecolor=bg_color, edgecolor=border_color, linewidth=1.4, zorder=1)
    ax.add_patch(card)
    # Title badge at top of card
    ax.text(x + w/2, y + h - 0.28, title, ha='center', va='center',
            fontsize=10.5, fontweight='bold', color=title_color, family='sans-serif', zorder=2)

def draw_box(ax, x, y, w, h, title, lines, bg_color, border_color, title_color='#0f172a', badge=None, badge_bg='#e2e8f0'):
    """Draws an inner processing module box"""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.1",
                         facecolor=bg_color, edgecolor=border_color, linewidth=1.2, zorder=3)
    ax.add_patch(box)
    
    # Title
    t_y = y + h - 0.22 if lines else y + h/2
    ax.text(x + w/2, t_y, title, ha='center', va='center',
            fontsize=9.2, fontweight='bold', color=title_color, family='sans-serif', zorder=4)
    
    # Text lines
    if lines:
        for i, line in enumerate(lines):
            l_y = t_y - 0.25 - (i * 0.21)
            ax.text(x + w/2, l_y, line, ha='center', va='center',
                    fontsize=8.0, color='#334155', family='sans-serif', zorder=4)
            
    # Optional corner badge
    if badge:
        b_w, b_h = 1.0, 0.22
        b_x, b_y = x + w - b_w - 0.08, y + 0.08
        badge_box = FancyBboxPatch((b_x, b_y), b_w, b_h, boxstyle="round,pad=0.02,rounding_size=0.05",
                                   facecolor=badge_bg, edgecolor='#94a3b8', linewidth=0.8, zorder=5)
        ax.add_patch(badge_box)
        ax.text(b_x + b_w/2, b_y + b_h/2, badge, ha='center', va='center',
                fontsize=7.2, fontweight='bold', color='#1e293b', family='sans-serif', zorder=6)

def draw_arrow(ax, x1, y1, x2, y2, color='#64748b', lw=1.6):
    """Draws a clean professional connection arrow"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle='-|>,head_length=5,head_width=3.5',
                            connectionstyle='arc3,rad=0',
                            color=color, linewidth=lw, zorder=7)
    ax.add_patch(arrow)

# ==============================================================================
# 4 Main Stages: Left-to-Right Horizontal Pipeline
# ==============================================================================

# Stage 1: Input & Linguistic Feature Extraction
draw_card(ax, 0.4, 0.4, 3.1, 6.4, "1. INPUT & LINGUISTIC PIPELINE", bg_color='#f8fafc', border_color='#94a3b8', title_color='#1e293b')

draw_box(ax, 0.65, 5.0, 2.6, 1.25, "Raw News Article",
         ["Cyber threat reports", "e.g., 'LockBit breached server", "exploiting CVE-2023-...'"],
         bg_color='#ffffff', border_color='#94a3b8')

draw_arrow(ax, 1.95, 5.0, 1.95, 4.45)

draw_box(ax, 0.65, 3.2, 2.6, 1.25, "spaCy POS Tagger",
         ["Universal Tagging (20 classes)", "NOUN, PROPN, VERB, NUM...", "Syntactic word categories"],
         bg_color='#f0fdf4', border_color='#22c55e', badge="20 Tags", badge_bg='#dcfce7')

draw_arrow(ax, 1.95, 3.2, 1.95, 2.65)

draw_box(ax, 0.65, 1.4, 2.6, 1.25, "spaCy General NER",
         ["Coarse Entities (4 classes)", "PERSON, ORG, GPE, LOC", "Anchors multi-word nouns"],
         bg_color='#eff6ff', border_color='#3b82f6', badge="4 Classes", badge_bg='#dbeafe')


# Connections Stage 1 -> Stage 2
draw_arrow(ax, 3.25, 5.62, 3.9, 5.62)  # Raw text to Tokenizer/Backbone
draw_arrow(ax, 3.25, 3.82, 3.9, 3.82)  # POS to POS Embedding
draw_arrow(ax, 3.25, 2.02, 3.9, 2.02)  # NER to NER Embedding


# Stage 2: Multi-Feature Transformer Encoding
draw_card(ax, 3.9, 0.4, 3.6, 6.4, "2. MULTI-FEATURE ENCODING", bg_color='#faf5ff', border_color='#c084fc', title_color='#6b21a8')

draw_box(ax, 4.15, 4.7, 3.1, 1.6, "Transformer Backbone",
         ["Subword Tokenizer + Encoder", "SecBERT / BERT / RoBERTa", "Yields contextual hidden state"],
         bg_color='#ffffff', border_color='#10b981', badge="Dim: 768", badge_bg='#d1fae5')

draw_box(ax, 4.15, 3.1, 3.1, 1.3, "POS Embedding Layer",
         ["Learnable Lookup Table", "Projects 20 POS IDs to Dense Vector"],
         bg_color='#ffffff', border_color='#22c55e', badge="Dim: 32", badge_bg='#dcfce7')

draw_box(ax, 4.15, 1.5, 3.1, 1.3, "NER Embedding Layer",
         ["Learnable Lookup Table", "Projects 4 Coarse NER IDs to Vector"],
         bg_color='#ffffff', border_color='#3b82f6', badge="Dim: 32", badge_bg='#dbeafe')

# Convergence into Concatenation Block
draw_arrow(ax, 7.25, 5.5, 7.85, 4.1)
draw_arrow(ax, 7.25, 3.75, 7.85, 3.6)
draw_arrow(ax, 7.25, 2.15, 7.85, 3.1)


# Stage 3: Feature Fusion & Cost-Sensitive Classification
draw_card(ax, 7.85, 0.4, 3.2, 6.4, "3. FUSION & CLASSIFICATION", bg_color='#fffbeb', border_color='#fcd34d', title_color='#b45309')

draw_box(ax, 8.05, 4.9, 2.8, 1.35, "Feature Concatenation",
         ["[ H_trans ; E_POS ; E_NER ]", "Concatenated representation", "Dense feature vector"],
         bg_color='#f3e8ff', border_color='#a855f7', badge="Dim: 832", badge_bg='#e9d5ff')

draw_arrow(ax, 9.45, 4.9, 9.45, 4.35)

draw_box(ax, 8.05, 3.0, 2.8, 1.35, "Classification Head",
         ["Dropout (p = 0.1)", "Linear Projection Layer", "Generates 63 class emission logits"],
         bg_color='#ffffff', border_color='#f59e0b', badge="63 Logits", badge_bg='#fef3c7')

draw_arrow(ax, 9.45, 3.0, 9.45, 2.45)

draw_box(ax, 8.05, 1.1, 2.8, 1.35, "Cost-Sensitive Loss",
         ["Weighted Cross-Entropy", "Downweights 'O' Tag (w_O = 0.35)", "Full penalty on 62 targets (w = 1.0)"],
         bg_color='#fef2f2', border_color='#ef4444', badge="w_O = 0.35", badge_bg='#fee2e2')

# Connection Stage 3 -> Stage 4
draw_arrow(ax, 10.85, 3.67, 11.45, 3.67)


# Stage 4: Constrained Decoding & Extraction Output
draw_card(ax, 11.45, 0.4, 2.65, 6.4, "4. DECODING & OUTPUT", bg_color='#f0fdfa', border_color='#5eead4', title_color='#0f766e')

draw_box(ax, 11.65, 4.9, 2.25, 1.35, "Multi-Model Ensemble",
         ["Word-Level Softmax Averaging", "RoBERTa + SecBERT + BERT", "Combines complementary backbones"],
         bg_color='#ffffff', border_color='#0d9488')

draw_arrow(ax, 12.77, 4.9, 12.77, 4.35)

draw_box(ax, 11.65, 3.0, 2.25, 1.35, "Viterbi BIO Decoder",
         ["63 x 63 Transition Constraints", "Eliminates syntax violations", "Optimal DP sequence search"],
         bg_color='#ffffff', border_color='#0284c7', badge="Matrix 63x63", badge_bg='#e0f2fe')

draw_arrow(ax, 12.77, 3.0, 12.77, 2.45)

# Final Output Spans Box
draw_box(ax, 11.65, 0.7, 2.25, 1.75, "Extracted Event Spans",
         ["[Event]: Ransomware", "[Attacker]: LockBit", "[Victim]: Financial Servers", "[Tool]: Ransomware script", "[CVE]: CVE-2023-38606"],
         bg_color='#f8fafc', border_color='#059669', title_color='#047857')

plt.tight_layout()

output_path = Path(r"d:\Kasetsart\NLP\LPLLL\figures\system_architecture.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(str(output_path), dpi=300, bbox_inches='tight', pad_inches=0.1)
print(f"Stunning system architecture diagram generated at: {output_path}")
