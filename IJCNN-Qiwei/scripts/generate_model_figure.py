#!/usr/bin/env python3
"""Generate a publication-style architecture figure for the current Qiwei model."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)


COLORS = {
    "input": "#D9EAF7",
    "adapter": "#E8DFF5",
    "world": "#FBE5D6",
    "trace": "#DDEFD9",
    "rag": "#FFF2CC",
    "ssm": "#E4D7F5",
    "transformer": "#D5E8D4",
    "output": "#F8CECC",
    "line": "#2B2B2B",
    "muted": "#666666",
    "panel": "#F8F9FB",
}


def box(ax, xy, wh, text, fc, ec="#2B2B2B", fontsize=9, weight="regular", radius=0.06):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.018,rounding_size={radius}",
        linewidth=1.15,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color="#111111",
        linespacing=1.18,
    )
    return patch


def panel(ax, xy, wh, title):
    x, y = xy
    w, h = wh
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            linewidth=0.9,
            edgecolor="#B7B7B7",
            facecolor=COLORS["panel"],
        )
    )
    ax.text(x + 0.16, y + h - 0.23, title, ha="left", va="top", fontsize=11, fontweight="bold")


def arrow(ax, start, end, text=None, rad=0.0, lw=1.2, color=None, fontsize=7.5):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=lw,
        color=color or COLORS["line"],
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(patch)
    if text:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2
        ax.text(mx, my + 0.08, text, ha="center", va="bottom", fontsize=fontsize, color=COLORS["muted"])


def mini_token(ax, x, y, label, fc, w=1.05, h=0.32):
    box(ax, (x, y), (w, h), label, fc, fontsize=7.2, radius=0.035)


def draw_panel_a(ax):
    panel(ax, (0.25, 4.95), (5.25, 4.45), "A. Modal-Abductive Input and World Construction")
    box(ax, (0.58, 8.55), (1.35, 0.48), "Premises\nNL / FOL", COLORS["input"], fontsize=8.3, weight="bold")
    box(ax, (2.08, 8.55), (1.35, 0.48), "Question\nStem", COLORS["input"], fontsize=8.3, weight="bold")
    box(ax, (3.58, 8.55), (1.35, 0.48), "Answer\nCandidates", COLORS["input"], fontsize=8.3, weight="bold")

    box(ax, (1.15, 7.70), (3.25, 0.55), "Type Adapter\nType 1: natural-language logic   |   Type 2: symbolic/FOL logic", COLORS["adapter"], fontsize=8.2, weight="bold")
    arrow(ax, (1.25, 8.55), (2.05, 8.25))
    arrow(ax, (2.75, 8.55), (2.75, 8.25))
    arrow(ax, (4.25, 8.55), (3.45, 8.25))

    box(ax, (0.65, 6.77), (1.62, 0.62), "BGE-RAG\nfailure-mode\nsemantic memory", COLORS["rag"], fontsize=8.1, weight="bold")
    box(ax, (2.72, 6.77), (2.22, 0.62), "Candidate-Specific\nPossible Worlds", COLORS["world"], fontsize=8.4, weight="bold")
    arrow(ax, (2.75, 7.70), (1.55, 7.39), text="retrieval")
    arrow(ax, (2.75, 7.70), (3.83, 7.39), text="world builder")

    world_labels = [
        ("W_s", "support"),
        ("W_c", "counter"),
        ("W_u", "uncertain"),
        ("W_r", "rule-chain"),
    ]
    for i, (symbol, label) in enumerate(world_labels):
        x = 2.75 + (i % 2) * 1.08
        y = 6.05 - (i // 2) * 0.55
        box(ax, (x, y), (0.85, 0.34), f"{symbol}\n{label}", COLORS["world"], fontsize=7.0)

    arrow(ax, (1.46, 6.77), (2.78, 6.30), text="RAG features", rad=-0.08)
    arrow(ax, (3.82, 6.77), (3.58, 6.42), text="instantiate", rad=0.04)

    box(ax, (0.65, 5.20), (4.30, 0.55), "World Transition Trace\nobserve -> propose -> test -> compare -> refute/backtrack -> update belief -> rank", COLORS["trace"], fontsize=8.0, weight="bold")
    arrow(ax, (3.80, 6.05), (2.88, 5.75))
    arrow(ax, (1.46, 6.77), (1.68, 5.75), rad=0.08)


def draw_panel_b(ax):
    panel(ax, (5.78, 4.95), (4.75, 4.45), "B. Proof-State Trace Tokens")
    y0 = 8.55
    labels = [
        ("observe", COLORS["input"]),
        ("select premise", COLORS["trace"]),
        ("apply rule", COLORS["world"]),
        ("infer fact", COLORS["world"]),
        ("compare option", COLORS["trace"]),
        ("detect conflict", COLORS["output"]),
        ("backtrack / suspend", COLORS["output"]),
        ("update belief", COLORS["ssm"]),
        ("rank", COLORS["transformer"]),
    ]
    for i, (label, fc) in enumerate(labels):
        x = 6.10 + (i % 3) * 1.35
        y = y0 - (i // 3) * 0.62
        mini_token(ax, x, y, label, fc, w=1.12, h=0.36)

    for row in range(3):
        y = y0 + 0.18 - row * 0.62
        arrow(ax, (7.22, y), (7.44, y), lw=0.9)
        arrow(ax, (8.57, y), (8.79, y), lw=0.9)
    arrow(ax, (9.35, 7.22), (6.08, 6.83), text="trace order", rad=0.18, lw=1.0)

    box(
        ax,
        (6.14, 5.55),
        (3.98, 0.86),
        "Trace feature vector per step\n[action one-hot, support, contradiction, proof support,\nintermediate support, counter margin, belief reward, BGE features]",
        "#FFFFFF",
        fontsize=8.0,
    )

    box(
        ax,
        (6.42, 6.65),
        (3.35, 0.42),
        "Reasoning objective: make attention operate over proof states, not only text similarity",
        COLORS["adapter"],
        fontsize=7.6,
        weight="bold",
    )
    arrow(ax, (8.10, 6.65), (8.10, 6.41), lw=1.0)


def draw_panel_c(ax):
    panel(ax, (0.25, 0.45), (10.28, 4.25), "C. Neural World Model and Reasoning Transformer")

    box(ax, (0.58, 3.70), (1.75, 0.52), "Candidate\nFeature Encoder", COLORS["adapter"], fontsize=8.0, weight="bold")
    box(ax, (0.58, 2.83), (1.75, 0.52), "Trace Tensor\nT x d", COLORS["trace"], fontsize=8.0, weight="bold")
    box(ax, (2.75, 2.90), (1.75, 1.05), "SSM / WM\nlatent transition\nstate h_t", COLORS["ssm"], fontsize=8.3, weight="bold")
    arrow(ax, (2.33, 3.96), (2.75, 3.72))
    arrow(ax, (2.33, 3.09), (2.75, 3.18))

    block_x = 4.95
    block_y = 2.45
    block_w = 2.35
    block_h = 1.95
    box(ax, (block_x, block_y), (block_w, block_h), "Proof-State Transformer Block x N", "#FFFFFF", fontsize=8.8, weight="bold")
    sub = [
        ("AdaLN", COLORS["adapter"]),
        ("Block-Wise SSM", COLORS["ssm"]),
        ("Scale + Residual", "#EFEFEF"),
        ("AdaLN", COLORS["adapter"]),
        ("RoPE Local Attention", COLORS["transformer"]),
        ("Scale + Residual", "#EFEFEF"),
        ("SwiGLU FFN", COLORS["transformer"]),
    ]
    for i, (label, fc) in enumerate(sub):
        box(ax, (block_x + 0.25, block_y + block_h - 0.43 - i * 0.235), (1.85, 0.18), label, fc, fontsize=6.3, radius=0.025)
    arrow(ax, (4.50, 3.43), (4.95, 3.43), text="candidate token + trace states")

    box(ax, (7.80, 3.33), (1.90, 0.50), "Candidate-Level\nTransformer", COLORS["transformer"], fontsize=8.0, weight="bold")
    box(ax, (7.80, 2.45), (1.90, 0.50), "Implicit Causal\nAttention Bias", COLORS["ssm"], fontsize=8.0, weight="bold")
    arrow(ax, (7.30, 3.42), (7.80, 3.58))
    arrow(ax, (7.30, 3.05), (7.80, 2.72))

    box(ax, (8.30, 1.25), (1.15, 0.50), "Softmax\np(answer)", COLORS["output"], fontsize=8.2, weight="bold")
    arrow(ax, (8.75, 2.45), (8.80, 1.75), text="scores")

    ax.text(
        0.58,
        0.82,
        "Training signal: supervised cross-entropy over candidates; optional consistency regularization. "
        "Reported current Type 1 validation accuracy: 0.82716. Type 2 modal branch: 0.842105.",
        ha="left",
        va="center",
        fontsize=8.0,
        color="#333333",
    )


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(14, 10), dpi=220)
    ax.set_xlim(0, 10.8)
    ax.set_ylim(0, 9.75)
    ax.axis("off")

    ax.text(
        0.25,
        9.58,
        "Qiwei Model: Modal-Abductive Proof-State World Model with SSM/Transformer Reasoning",
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        0.25,
        9.28,
        "A unified reasoning architecture for Type 1 natural-language logic and Type 2 symbolic logic.",
        ha="left",
        va="top",
        fontsize=9,
        color=COLORS["muted"],
    )

    draw_panel_a(ax)
    draw_panel_b(ax)
    draw_panel_c(ax)

    # Cross-panel flow arrows.
    arrow(ax, (4.95, 5.48), (5.78, 5.98), text="proof-state tensor", rad=-0.10, lw=1.1)
    arrow(ax, (8.10, 5.55), (3.62, 4.00), text="encoded trace to WM/Transformer", rad=0.08, lw=1.1)

    caption = (
        "Figure. The current Qiwei model converts each candidate answer into a set of possible worlds "
        "and proof-state transition tokens. BGE-RAG retrieves failure-mode semantics, SSM/WM encodes "
        "latent state transitions, and a RoPE local-attention Transformer compares candidate proof states "
        "to produce the final answer distribution."
    )
    ax.text(0.25, 0.18, caption, ha="left", va="bottom", fontsize=7.6, color="#333333", wrap=True)

    fig.tight_layout(pad=0.1)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(OUT_DIR / f"qiwei_model_architecture.{ext}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
