#!/usr/bin/env python3
"""
PCA analysis of AF3 structural outputs for ER mutation comparison.

Compares WT, H524L, and Y537S dimer structures (holo/apo) and WT monomer
by projecting CA-coordinate vectors onto principal components after
Kabsch superposition onto a crystallographic reference structure.

Usage:
    python af3_pca.py [--reference PDB] [--top_n N] [--chains {AB,A}] [--out_dir DIR]

    --reference : Reference PDB for superposition (default: inputs/1A52_clean_dimer.pdb).
                  Use inputs/1A52_clean_apo.pdb with --chains A for monomer runs.
    --top_n     : Use only the top N structures per condition by ranking score
                  (default: all 1250)
    --chains    : 'AB' for full dimer (default), 'A' to use chain A only
                  (use 'A' to include monomer in the same PCA)
    --out_dir   : Where to write output plots (default: outputs/analysis/pca)
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ── optional deps with clear error messages ──────────────────────────────────
try:
    from sklearn.decomposition import PCA
except ImportError:
    sys.exit("sklearn not found. Install with: pip install scikit-learn")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    sys.exit("matplotlib not found. Install with: pip install matplotlib")


# ── paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS   = REPO_ROOT / "outputs"
INPUTS    = REPO_ROOT / "inputs"

# (label, condition_dir, is_dimer)
DATASETS = [
    ("WT dimer holo",    OUTPUTS / "af3_wt"         / "1a52_wt_holo",         True),
    ("WT dimer apo",     OUTPUTS / "af3_wt"         / "1a52_wt_apo",          True),
    ("H524L dimer holo", OUTPUTS / "af3_H524L"      / "1a52_h524l_holo",      True),
    ("H524L dimer apo",  OUTPUTS / "af3_H524L"      / "1a52_h524l_apo",       True),
    ("Y537S dimer holo", OUTPUTS / "af3_Y537S"      / "1a52_y537s_holo",      True),
    ("Y537S dimer apo",  OUTPUTS / "af3_Y537S"      / "1a52_y537s_apo",       True),
    ("WT mono holo",     OUTPUTS / "af3_wt_monomer" / "1a52_wt_monomer_holo", False),
    ("WT mono apo",      OUTPUTS / "af3_wt_monomer" / "1a52_wt_monomer_apo",  False),
]

# colour and marker per condition label
STYLE = {
    "WT dimer holo":    ("#1f77b4", "o"),
    "WT dimer apo":     ("#1f77b4", "^"),
    "H524L dimer holo": ("#d62728", "o"),
    "H524L dimer apo":  ("#d62728", "^"),
    "Y537S dimer holo": ("#2ca02c", "o"),
    "Y537S dimer apo":  ("#2ca02c", "^"),
    "WT mono holo":     ("#ff7f0e", "o"),
    "WT mono apo":      ("#ff7f0e", "^"),
}


# ── PDB parsing ──────────────────────────────────────────────────────────────
def parse_pdb_ca_coords(pdb_path: Path, chains: list[str]) -> np.ndarray:
    """
    Parse CA coords from a PDB file. Keeps only the first alternate conformation
    (blank or 'A'). Trims all chains to the shared residue-number range so
    asymmetric N/C-terminal overhangs between chains don't cause size mismatches.
    Returns a flat 1-D array ordered by chain then residue number.
    Raises ValueError if a requested chain is absent.
    """
    # first pass: collect (resnum -> xyz) per chain
    chain_coords: dict[str, dict[int, np.ndarray]] = {ch: {} for ch in chains}

    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            if line[16] not in (" ", "A"):          # skip alt conformation B, C, …
                continue
            chain = line[21]
            if chain not in chains:
                continue
            resnum = int(line[22:26])
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            chain_coords[chain][resnum] = np.array([x, y, z])

    for ch in chains:
        if not chain_coords[ch]:
            raise ValueError(f"Chain '{ch}' not found in {pdb_path}")

    # restrict to residue numbers present in every requested chain
    common = set.intersection(*(set(chain_coords[ch]) for ch in chains))
    common_sorted = sorted(common)

    segments = []
    for ch in chains:
        segments.append(np.array([chain_coords[ch][r] for r in common_sorted]))

    return np.concatenate([s.flatten() for s in segments])


# ── CIF parsing ──────────────────────────────────────────────────────────────
def parse_cif_ca_coords(cif_path: Path, chains: list[str]) -> np.ndarray | None:
    """
    Parse CA coords from an mmCIF file.
    Returns a flat 1-D array (chain A residues then chain B), or None if a
    requested chain is missing.
    """
    chain_coords: dict[str, dict[int, np.ndarray]] = {}
    in_atom = False
    col: dict[str, int] = {}

    with open(cif_path) as fh:
        for line in fh:
            stripped = line.strip()

            if stripped == "loop_":
                in_atom = False
                col = {}
                continue

            if stripped.startswith("_atom_site."):
                col[stripped.split(".")[-1]] = len(col)
                in_atom = True
                continue

            if not in_atom:
                continue
            if stripped.startswith("#"):
                in_atom = False
                continue
            if not stripped.startswith("ATOM"):
                continue

            parts = stripped.split()
            if len(parts) <= max(col.values(), default=0):
                continue

            if parts[col["label_atom_id"]] != "CA":
                continue

            chain = parts[col["auth_asym_id"]]
            if chain not in chains:
                continue

            seq_id = int(parts[col["label_seq_id"]])
            x = float(parts[col["Cartn_x"]])
            y = float(parts[col["Cartn_y"]])
            z = float(parts[col["Cartn_z"]])
            chain_coords.setdefault(chain, {})[seq_id] = np.array([x, y, z])

    segments = []
    for ch in chains:
        if ch not in chain_coords:
            return None
        res = chain_coords[ch]
        segments.append(np.array([res[r] for r in sorted(res)]))

    return np.concatenate([s.flatten() for s in segments])


# ── Kabsch superposition ──────────────────────────────────────────────────────
def kabsch(mobile: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Superimpose mobile (N×3) onto ref (N×3), return rotated mobile (N×3)."""
    mob_c = mobile.mean(0)
    ref_c = ref.mean(0)
    mob = mobile - mob_c
    ref = ref   - ref_c
    H = mob.T @ ref
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return mob @ R.T + ref_c


def superimpose_flat(mobile_flat: np.ndarray, ref_flat: np.ndarray) -> np.ndarray:
    n = len(mobile_flat) // 3
    rotated = kabsch(mobile_flat.reshape(n, 3), ref_flat.reshape(n, 3))
    return rotated.flatten()


# ── ranking helper ────────────────────────────────────────────────────────────
def ranked_sample_dirs(condition_dir: Path, top_n: int | None) -> list[Path]:
    """Return seed-sample dirs sorted by ranking_score descending, optionally capped."""
    df = pd.read_csv(condition_dir / "ranking_scores.csv")
    df = df.sort_values("ranking_score", ascending=False)
    if top_n is not None:
        df = df.head(top_n)
    dirs = []
    for _, row in df.iterrows():
        d = condition_dir / f"seed-{int(row['seed'])}_sample-{int(row['sample'])}"
        if d.exists():
            dirs.append(d)
    return dirs


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path,
                        default=INPUTS / "1A52_clean_dimer.pdb",
                        help="Reference PDB for Kabsch superposition")
    parser.add_argument("--top_n",   type=int, default=None,
                        help="Use only top N structures per condition (default: all)")
    parser.add_argument("--chains",  choices=["AB", "A"], default="AB",
                        help="'AB' = full dimer (default); 'A' = chain A only (enables monomer)")
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="Only include these conditions (default: all). "
                             "Example: --conditions 'WT mono holo' 'WT mono apo'")
    parser.add_argument("--out_dir", type=Path,
                        default=OUTPUTS / "analysis" / "pca",
                        help="Output directory for plots and CSV")
    args = parser.parse_args()

    chain_list      = list(args.chains)        # ['A','B'] or ['A']
    include_monomer = (args.chains == "A")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── load reference ───────────────────────────────────────────────────────
    if not args.reference.exists():
        sys.exit(f"Reference PDB not found: {args.reference}")

    print(f"Reference : {args.reference}")
    print(f"Chains    : {args.chains}  |  top_n: {args.top_n or 'all'}")
    print(f"Output    : {args.out_dir}\n")

    try:
        ref_flat = parse_pdb_ca_coords(args.reference, chain_list)
    except ValueError as e:
        sys.exit(f"Error reading reference: {e}")

    n_ref = len(ref_flat) // 3
    print(f"Reference CA atoms: {n_ref} ({len(ref_flat)} coords)\n")

    # ── collect all vectors ──────────────────────────────────────────────────
    all_vecs: list[np.ndarray] = []
    all_meta: list[dict] = []               # seed, sample, cif_path per structure
    condition_indices: dict[str, list[int]] = {}

    datasets = [d for d in DATASETS if include_monomer or d[2]]
    if args.conditions:
        datasets = [d for d in datasets if d[0] in args.conditions]
        if not datasets:
            sys.exit(f"No datasets match --conditions {args.conditions}")

    for label, cond_dir, is_dimer in datasets:
        if not cond_dir.exists():
            print(f"  [SKIP] {label}: directory not found ({cond_dir})")
            continue

        sample_dirs = ranked_sample_dirs(cond_dir, args.top_n)
        print(f"  Loading {label} ({len(sample_dirs)} structures) …", flush=True)

        # monomers only have chain A regardless of --chains
        chains = chain_list if is_dimer else ["A"]
        vecs = []
        metas = []

        for sd in sample_dirs:
            cif = sd / "model.cif"
            if not cif.exists():
                continue
            flat = parse_cif_ca_coords(cif, chains)
            if flat is None:
                continue

            if len(flat) != len(ref_flat):
                if not vecs:
                    print(f"    [WARN] {label}: CA count {len(flat)//3} != ref {n_ref}; skipping all")
                break

            # parse seed and sample from directory name (seed-N_sample-M)
            parts = sd.name.split("_")
            seed   = int(parts[0].split("-")[1])
            sample = int(parts[1].split("-")[1])

            vecs.append(superimpose_flat(flat, ref_flat))
            metas.append({"seed": seed, "sample": sample, "cif_path": str(cif)})

        if not vecs:
            print(f"    [WARN] no valid structures for {label}")
            continue

        idx_start = len(all_vecs)
        all_vecs.extend(vecs)
        all_meta.extend(metas)
        condition_indices[label] = list(range(idx_start, idx_start + len(vecs)))
        print(f"    → {len(vecs)} structures loaded")

    if not all_vecs:
        sys.exit("No structures loaded — check directory paths and reference chain count.")

    X = np.array(all_vecs)
    print(f"\nTotal structures : {X.shape[0]}")
    print(f"Feature dimension: {X.shape[1]} (CA coords)")

    # ── PCA ──────────────────────────────────────────────────────────────────
    print("Running PCA …", flush=True)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_ * 100

    print("\nExplained variance:")
    for i, ev in enumerate(explained, 1):
        print(f"  PC{i}: {ev:.2f}%")

    # ── save PC scores ────────────────────────────────────────────────────────
    rows = []
    for label, idxs in condition_indices.items():
        for rank, idx in enumerate(idxs, 1):
            row = {"condition": label, "structure_rank": rank,
                   "seed": all_meta[idx]["seed"],
                   "sample": all_meta[idx]["sample"],
                   "cif_path": all_meta[idx]["cif_path"],
                   "PC1": X_pca[idx, 0], "PC2": X_pca[idx, 1]}
            rows.append(row)
    csv_path = args.out_dir / "pca_scores.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nPC scores saved → {csv_path}")

    # ── plots — one per variant, shared axis limits ───────────────────────────
    ref_name = args.reference.stem

    # group conditions by variant
    VARIANT_GROUPS = {
        "WT dimer":   ["WT dimer holo",    "WT dimer apo"],
        "H524L":      ["H524L dimer holo", "H524L dimer apo"],
        "Y537S":      ["Y537S dimer holo", "Y537S dimer apo"],
        "WT monomer": ["WT mono holo",     "WT mono apo"],
    }

    # compute shared axis limits across all structures
    x_all = X_pca[:, 0]
    y_all = X_pca[:, 1]
    pad = 0.05
    x_margin = (x_all.max() - x_all.min()) * pad
    y_margin = (y_all.max() - y_all.min()) * pad
    xlim = (x_all.min() - x_margin, x_all.max() + x_margin)
    ylim = (y_all.min() - y_margin, y_all.max() + y_margin)

    for variant, labels in VARIANT_GROUPS.items():
        # only plot variants that have data
        present = [lbl for lbl in labels if lbl in condition_indices]
        if not present:
            continue

        fig, ax = plt.subplots(figsize=(7, 6))
        HOLO_COLOR = "#1f77b4"   # blue
        APO_COLOR  = "#d62728"   # red

        for lbl in present:
            pts = X_pca[condition_indices[lbl]]
            is_holo = "holo" in lbl
            color = HOLO_COLOR if is_holo else APO_COLOR
            kind  = "holo"    if is_holo else "apo"
            ax.scatter(pts[:, 0], pts[:, 1],
                       c=color, marker="o", s=8, alpha=0.3, linewidths=0,
                       label=kind)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)", fontsize=11)
        ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)", fontsize=11)
        ax.set_title(f"{variant} — PC1 vs PC2\n(CA coords, superposed onto {ref_name})", fontsize=12)
        ax.legend(fontsize=9, loc="best", framealpha=0.7)
        plt.tight_layout()

        fname = variant.lower().replace(" ", "_") + "_pca.png"
        p = args.out_dir / fname
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved → {p}")

    print("\nDone.")


if __name__ == "__main__":
    main()
