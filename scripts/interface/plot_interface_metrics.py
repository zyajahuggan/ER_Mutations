#!/usr/bin/env python3
"""
plot_interface_metrics.py

Heatmaps for the four interface-quality metrics that interface_analyzer.py /
interface_analyzer_Y537.py already compute into dms_H524_interface_scores.csv /
dms_Y537_interface_scores.csv alongside dG_interface, but never got their own
figure (only ddG_complex/ddG_interface are plotted there):

  ddSASA         buried interface surface area
  dsc            shape complementarity
  dpackstat      interface packing quality
  dunsat_hbonds  buried unsatisfied interface polar atoms

All four are relative to that condition's own WT baseline, same as ddG. For
each (condition, mutant) cell, the replicate is chosen by the lowest raw
dG_interface -- the same selection rule interface_analyzer.py's ddG heatmap
uses -- so every panel here refers to the same representative structure per
cell as the existing ddG figure, and the four panels stay comparable to it.

dSASA/sc/packstat are "bigger is better" for the interface, so their panels
use a colormap where positive (improved) reads blue and negative (worse)
reads red. unsat_hbonds is "bigger is worse" (more buried unsatisfied polar
atoms), so its panel's colormap is flipped so that positive (worse) still
reads red -- i.e. red consistently means "interface got worse" across all
four panels, matching the convention in interface_analyzer.py's ddG heatmap
(positive ddG = destabilizing = red).

Works unchanged on either CSV -- the mutation site (e.g. H524 or Y537) is
parsed from the "mutation" column rather than hardcoded, since both CSVs
share the same schema.

Usage:
    python3 plot_interface_metrics.py --csv ../outputs/dms_H524_interface_scores.csv \
                                       --out_fig ../outputs/dms_H524_interface_metrics.png
    python3 plot_interface_metrics.py --csv ../outputs/dms_Y537_interface_scores.csv \
                                       --out_fig ../outputs/dms_Y537_interface_metrics.png
"""
import argparse
import re
import sys

# (delta column, raw column, display label, higher_is_better)
METRICS = [
    ("ddSASA",        "dSASA",        "ΔdSASA (Å²)\nburied interface area",              True),
    ("dsc",           "sc",           "Δsc\nshape complementarity",                       True),
    ("dpackstat",     "packstat",     "Δpackstat\ninterface packing quality",              True),
    ("dunsat_hbonds", "unsat_hbonds", "Δ unsat H-bonds\nburied unsatisfied polar atoms",   False),
]

# Conditions with a real interface (mono_apo is a single chain, no ligand -> skip)
INTERFACE_CONDITIONS = ("dimer_apo", "mono_holo", "dimer_holo")


def _select_best_rows(df, rows, aas):
    """For each (condition, mut_aa), pick the replicate with the lowest raw
    dG_interface -- same convention as interface_analyzer.py's ddG heatmap --
    so every metric panel here refers to the same representative structure per
    cell, and stays comparable to the existing ddG_interface figure."""
    best = {}
    for cond_name in rows:
        sub = df[df["condition"] == cond_name]
        for aa in aas:
            aa_rows = sub[sub["mut_aa"] == aa].dropna(subset=["dG_interface"])
            if not aa_rows.empty:
                best[(cond_name, aa)] = aa_rows.loc[aa_rows["dG_interface"].idxmin()]
    return best


def _panel(ax, fig, best, aas, rows, delta_col, title, higher_is_better):
    import matplotlib
    import numpy as np

    matrix = []
    for cond_name in rows:
        vals = []
        for aa in aas:
            row = best.get((cond_name, aa))
            v = row[delta_col] if row is not None and row[delta_col] != "" else float("nan")
            vals.append(float(v) if v is not None else float("nan"))
        matrix.append(vals)
    matrix = np.array(matrix, dtype=float)

    finite = matrix[~np.isnan(matrix)]
    if finite.size:
        vmax = np.percentile(np.abs(finite), 90)
        vmax = vmax if vmax > 1e-9 else np.max(np.abs(finite))
    else:
        vmax = 1.0
    vmax = vmax if vmax > 1e-9 else 1.0
    norm = matplotlib.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    # Keep "red = interface got worse" consistent across all four panels: for
    # bigger-is-better metrics (dSASA, sc, packstat) positive should read blue,
    # so use the un-reversed colormap; unsat_hbonds is bigger-is-worse, so its
    # panel uses the reversed map instead (positive still lands on red).
    cmap = "RdBu" if higher_is_better else "RdBu_r"

    clipped_hi = bool(np.any(finite > vmax)) if finite.size else False
    clipped_lo = bool(np.any(finite < -vmax)) if finite.size else False
    extend = "both" if clipped_hi and clipped_lo else "max" if clipped_hi else "min" if clipped_lo else "neither"

    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(aas)))
    ax.set_xticklabels(aas)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_title(title, fontsize=10, loc="left")

    ax.set_xticks(np.arange(-0.5, len(aas), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    holo_idx = next((i for i, name in enumerate(rows) if "holo" in name), None)
    if holo_idx is not None and holo_idx > 0:
        ax.axhline(holo_idx - 0.5, color="black", linewidth=1.5)

    for i in range(len(rows)):
        for j in range(len(aas)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=6.5, color="black" if abs(val) < 0.6 * vmax else "white")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, extend=extend)
    cbar.set_label(delta_col, fontsize=7)


def plot_metrics(csv_path, out_fig):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        sys.exit("matplotlib/pandas not found. Install with: pip install matplotlib pandas")

    df = pd.read_csv(csv_path)
    df = df[~df["is_wt"]]  # WT (delta=0 by construction) doesn't need its own column
    if df.empty:
        sys.exit(f"No non-WT rows in {csv_path}")

    aas = sorted(df["mut_aa"].unique())

    m = re.match(r"^([A-Z])(\d+)", str(df["mutation"].iloc[0]))
    site_label = f"{m.group(1)}{m.group(2)}" if m else "site"

    interface_rows = [c for c in INTERFACE_CONDITIONS if c in df["condition"].unique()]
    if not interface_rows:
        sys.exit(f"No interface conditions ({INTERFACE_CONDITIONS}) found in {csv_path}")

    best = _select_best_rows(df, interface_rows, aas)

    fig, axes = plt.subplots(
        len(METRICS), 1,
        figsize=(0.55 * len(aas) + 2.2, 1.5 * len(interface_rows) * len(METRICS) * 0.55 + 1.5),
    )

    for ax, (delta_col, raw_col, title, higher_is_better) in zip(axes, METRICS):
        _panel(ax, fig, best, aas, interface_rows, delta_col, title, higher_is_better)

    for ax in axes[:-1]:
        ax.set_xticklabels([])
    axes[-1].set_xlabel(f"Mutant residue at {site_label}")

    fig.suptitle(f"ERα {site_label} DMS — interface quality metrics vs. WT "
                 "(lowest-dG_interface replicate per mutant)", y=1.01)
    fig.tight_layout()
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    print(f"Heatmap written to {out_fig}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot dSASA/sc/packstat/unsat_hbonds ΔΔ heatmaps from an "
                     "interface_analyzer.py-style scores CSV (H524 or Y537)"
    )
    parser.add_argument("--csv", required=True,
                         help="e.g. ../outputs/dms_H524_interface_scores.csv or dms_Y537_interface_scores.csv")
    parser.add_argument("--out_fig", required=True)
    args = parser.parse_args()
    plot_metrics(args.csv, args.out_fig)


if __name__ == "__main__":
    main()
