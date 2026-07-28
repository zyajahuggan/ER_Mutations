#!/usr/bin/env python3
"""
plot_coactivator_scores.py

Grouped bar chart of ddG_interface (Rosetta REU) from coactivator_scores.csv:
one group per mutation, bars for apo vs holo (coactivator-bound), each mutant
plotted relative to its own WT baseline (0).

Usage:
    conda activate pyrosetta
    python3 plot_coactivator_scores.py \
        --csv ../outputs/coactivator_scores.csv \
        --out ../outputs/coactivator_ddG_interface.png
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

APO_COLOR = "#2a78d6"
HOLO_COLOR = "#eb6834"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(os.path.dirname(__file__), "..", "outputs", "coactivator_scores.csv"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "outputs", "coactivator_ddG_interface.png"))
    ap.add_argument("--show", action="store_true", help="also open an interactive window")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    mutants = df[~df["is_wt"]].copy()

    labels = sorted(mutants["label"].unique())
    conditions = ["apo", "holo"]
    n_groups = len(labels)
    bar_w = 0.32
    x = range(n_groups)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for i, cond in enumerate(conditions):
        color = APO_COLOR if cond == "apo" else HOLO_COLOR
        offset = (i - 0.5) * bar_w
        vals = []
        mut_names = []
        for label in labels:
            row = mutants[(mutants["label"] == label) & (mutants["condition"] == cond)]
            vals.append(row["ddG_interface"].iloc[0])
            mut_names.append(row["mutation"].iloc[0])
        xs = [xi + offset for xi in x]
        bars = ax.bar(xs, vals, width=bar_w, color=color, label=cond.capitalize(),
                      edgecolor="none", zorder=3)
        for bx, v in zip(xs, vals):
            ax.text(bx, v + (0.3 if v >= 0 else -0.3), f"{v:+.1f}",
                    ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=9, fontweight="bold", color="#0b0b0b")

    ax.axhline(0, color="#c3c2b7", linewidth=1.2, zorder=2)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{label}\n({mutants[mutants['label']==label]['mutation'].iloc[0]})" for label in labels])
    ax.set_ylabel("ΔΔG interface (REU)\nvs. own WT")
    ax.set_title("Coactivator interface binding: effect of ER mutations")
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
