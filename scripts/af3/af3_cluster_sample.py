#!/usr/bin/env python3
"""
Cluster AF3 structures per condition in PCA space and extract one representative
CIF per cluster.

Each condition (WT holo, H524L apo, etc.) is clustered independently in the
shared PC1/PC2 space so each can have its own optimal number of clusters.
The representative for each cluster is the structure closest to the centroid.

Usage:
    python af3_cluster_sample.py [--k N] [--k_range MIN MAX] [--scores CSV]
                                 [--out_dir DIR] [--conditions ...]

    --k           : Use this fixed k for every condition. If omitted, k is chosen
                    per condition via silhouette analysis over --k_range.
    --k_range     : Min and max k to search (default: 2 8)
    --scores      : Path to pca_scores.csv (default: outputs/analysis/pca/pca_scores.csv)
    --out_dir     : Output directory (default: outputs/analysis/pca/cluster_reps)
    --conditions  : Conditions to include (default: all).
                    Example: --conditions "WT dimer holo" "H524L dimer holo"
"""

import sys
import shutil
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
except ImportError:
    sys.exit("sklearn not found. Install with: pip install scikit-learn")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    sys.exit("matplotlib not found. Install with: pip install matplotlib")


REPO_ROOT    = Path(__file__).resolve().parent.parent
DEFAULT_SCORES = REPO_ROOT / "outputs" / "analysis" / "pca" / "pca_scores.csv"
DEFAULT_OUT    = REPO_ROOT / "outputs" / "analysis" / "pca" / "cluster_reps"

CONDITION_COLORS = {
    "WT dimer holo":    "#1f77b4",
    "WT dimer apo":     "#aec7e8",
    "H524L dimer holo": "#d62728",
    "H524L dimer apo":  "#f7b6b2",
    "Y537S dimer holo": "#2ca02c",
    "Y537S dimer apo":  "#98df8a",
    "WT mono holo":     "#ff7f0e",
    "WT mono apo":      "#ffbb78",
}


def best_k_silhouette(X: np.ndarray, k_min: int, k_max: int) -> int:
    """Return the k in [k_min, k_max] with the highest silhouette score."""
    if len(X) <= k_min:
        return 1
    best_k, best_score = k_min, -1.0
    for k in range(k_min, min(k_max + 1, len(X))):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
        score  = silhouette_score(X, labels)
        if score > best_score:
            best_score, best_k = score, k
    return best_k


def cluster_condition(X: np.ndarray, k: int):
    """Run K-means and return (labels, centroids)."""
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    return labels, km.cluster_centers_


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k",          type=int, default=None,
                        help="Fixed k for all conditions (default: auto per condition)")
    parser.add_argument("--k_range",    type=int, nargs=2, default=[2, 8],
                        metavar=("MIN", "MAX"),
                        help="K range for silhouette search (default: 2 8)")
    parser.add_argument("--scores",     type=Path, default=DEFAULT_SCORES,
                        help="pca_scores.csv from af3_pca.py")
    parser.add_argument("--out_dir",    type=Path, default=DEFAULT_OUT,
                        help="Output directory for representative CIFs and plots")
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="Conditions to include (default: all)")
    args = parser.parse_args()

    if not args.scores.exists():
        sys.exit(f"Scores file not found: {args.scores}\nRun af3_pca.py first.")

    df = pd.read_csv(args.scores)

    if args.conditions:
        df = df[df["condition"].isin(args.conditions)]
        if df.empty:
            sys.exit(f"No rows match conditions: {args.conditions}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    k_min, k_max = args.k_range
    all_reps = []

    # shared axis limits for plots
    xlim = (df["PC1"].min(), df["PC1"].max())
    ylim = (df["PC2"].min(), df["PC2"].max())
    pad  = 0.05
    xlim = (xlim[0] - abs(xlim[0]) * pad, xlim[1] + abs(xlim[1]) * pad)
    ylim = (ylim[0] - abs(ylim[0]) * pad, ylim[1] + abs(ylim[1]) * pad)

    conditions = sorted(df["condition"].unique())
    n_conds    = len(conditions)
    ncols      = min(n_conds, 3)
    nrows      = (n_conds + ncols - 1) // ncols
    fig, axes  = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows),
                               squeeze=False)

    for ax_idx, cond in enumerate(conditions):
        cond_df = df[df["condition"] == cond].copy()
        X       = cond_df[["PC1", "PC2"]].values

        # choose k
        if args.k is not None:
            k = args.k
            method = f"k={k} (fixed)"
        else:
            k = best_k_silhouette(X, k_min, k_max)
            method = f"k={k} (silhouette)"

        print(f"\n{cond}  [{method}, n={len(cond_df)}]")

        if k == 1:
            labels    = np.zeros(len(X), dtype=int)
            centroids = X.mean(0, keepdims=True)
        else:
            labels, centroids = cluster_condition(X, k)

        cond_df = cond_df.copy()
        cond_df["cluster"] = labels

        cluster_colors = plt.cm.tab10(np.linspace(0, 1, max(k, 2)))

        # ── sample representative per cluster ─────────────────────────────────
        cond_tag = cond.lower().replace(" ", "_")

        for c in range(k):
            members  = cond_df[cond_df["cluster"] == c]
            centroid = centroids[c]
            dists    = np.linalg.norm(members[["PC1", "PC2"]].values - centroid, axis=1)
            rep      = members.iloc[np.argmin(dists)]

            src = Path(rep["cif_path"])
            if not src.exists():
                print(f"  [WARN] cluster {c}: CIF not found at {src}")
                continue

            dest = args.out_dir / f"{cond_tag}_cluster_{c}_rep.cif"
            shutil.copy(src, dest)

            all_reps.append({
                "condition":        cond,
                "cluster":          c,
                "k":                k,
                "seed":             int(rep["seed"]),
                "sample":           int(rep["sample"]),
                "PC1":              rep["PC1"],
                "PC2":              rep["PC2"],
                "dist_to_centroid": float(dists.min()),
                "cluster_size":     int(len(members)),
                "cif_path":         str(dest),
            })
            print(f"  Cluster {c} (n={len(members)}): "
                  f"seed-{int(rep['seed'])}_sample-{int(rep['sample'])}  → {dest.name}")

        # ── subplot ───────────────────────────────────────────────────────────
        ax = axes[ax_idx // ncols][ax_idx % ncols]
        color = CONDITION_COLORS.get(cond, "#888888")

        for c in range(k):
            members = cond_df[cond_df["cluster"] == c]
            ax.scatter(members["PC1"], members["PC2"],
                       color=cluster_colors[c], s=6, alpha=0.35,
                       linewidths=0, label=f"Cluster {c} (n={len(members)})")

        # centroids
        ax.scatter(centroids[:, 0], centroids[:, 1],
                   c="black", marker="x", s=60, linewidths=1.5, zorder=5)

        # rep stars
        for rep_row in [r for r in all_reps if r["condition"] == cond]:
            c = rep_row["cluster"]
            ax.scatter(rep_row["PC1"], rep_row["PC2"],
                       color=cluster_colors[c], marker="*", s=200,
                       edgecolors="black", linewidths=0.5, zorder=6)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f"{cond}\n{method}", fontsize=9)
        ax.set_xlabel("PC1", fontsize=8)
        ax.set_ylabel("PC2", fontsize=8)
        ax.legend(fontsize=7, loc="best", framealpha=0.6)

    # hide unused axes
    for ax_idx in range(n_conds, nrows * ncols):
        axes[ax_idx // ncols][ax_idx % ncols].set_visible(False)

    fig.suptitle("Per-condition clustering in shared PCA space\n★ = representative  × = centroid",
                 fontsize=12)
    plt.tight_layout()
    plot_path = args.out_dir / "clusters.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved → {plot_path}")

    # ── save summary CSV ──────────────────────────────────────────────────────
    rep_csv = args.out_dir / "cluster_representatives.csv"
    pd.DataFrame(all_reps).to_csv(rep_csv, index=False)
    print(f"Representatives saved → {rep_csv}")
    print("\nDone.")


if __name__ == "__main__":
    main()
