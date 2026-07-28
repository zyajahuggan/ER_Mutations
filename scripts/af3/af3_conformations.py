#!/usr/bin/env python3
"""
Find distinct conformations within each AF3 condition independently.

For each condition (e.g. "WT dimer holo"), runs PCA on just that condition's
structures, finds the optimal number of clusters via silhouette analysis, and
extracts the representative CIF closest to each cluster centroid.

Usage (from the scripts/ directory):
    python af3_conformations.py [--k_range MIN MAX] [--top_n N] [--out_dir DIR]

    --k_range : k values to search per condition (default: 2 8)
    --top_n   : use only top N structures per condition by ranking score (default: all)
    --out_dir : output directory (default: ../outputs/analysis/conformations)
"""

import sys
import shutil
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool

try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors
except ImportError:
    sys.exit("sklearn not found.  pip install scikit-learn")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib not found.  pip install matplotlib")


# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS   = REPO_ROOT / "outputs"
INPUTS    = REPO_ROOT / "inputs"

CONDITIONS = [
    # (label,            condition_dir,                                   reference_pdb,                    chains,      file_prefix)
    ("WT dimer holo",    OUTPUTS/"af3_wt"/"1a52_wt_holo",                INPUTS/"1A52_clean_dimer.pdb",    ["A","B"],  "wtdimer"),
    ("WT dimer apo",     OUTPUTS/"af3_wt"/"1a52_wt_apo",                 INPUTS/"1A52_clean_dimer.pdb",    ["A","B"],  "wtapo"),
    ("H524L dimer holo", OUTPUTS/"af3_H524L"/"1a52_h524l_holo",          INPUTS/"1A52_clean_dimer.pdb",    ["A","B"],  "h524lholo"),
    ("H524L dimer apo",  OUTPUTS/"af3_H524L"/"1a52_h524l_apo",           INPUTS/"1A52_clean_dimer.pdb",    ["A","B"],  "h524lapo"),
    ("Y537S dimer holo", OUTPUTS/"af3_Y537S"/"1a52_y537s_holo",          INPUTS/"1A52_clean_dimer.pdb",    ["A","B"],  "y537sholo"),
    ("Y537S dimer apo",  OUTPUTS/"af3_Y537S"/"1a52_y537s_apo",           INPUTS/"1A52_clean_dimer.pdb",    ["A","B"],  "y537sapo"),
    ("WT mono holo",     OUTPUTS/"af3_wt_monomer"/"1a52_wt_monomer_holo", INPUTS/"1A52_clean_apo.pdb",     ["A"],       ""),
    ("WT mono apo",      OUTPUTS/"af3_wt_monomer"/"1a52_wt_monomer_apo",  INPUTS/"1A52_clean_apo.pdb",     ["A"],       ""),
]


# ── PDB parser ────────────────────────────────────────────────────────────────
def parse_pdb_ca(pdb_path: Path, chains: list[str]) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """Returns (flat CA coords, residue_order) where residue_order[i] = (chain, position)
    for the i-th CA atom in the flattened array (3 floats each). `position` is a
    1-based positional index into the common residue set (reset per chain), NOT the
    reference PDB's native residue number — this matches AF3's CIF numbering, which
    renumbers each chain starting at 1 regardless of the crystal structure's native
    numbering. All CIF models are assumed to share this same residue ordering, so
    residue_order (from the reference structure) can be reused to index into any
    superimposed model vector."""
    chain_coords: dict[str, dict[int, np.ndarray]] = {ch: {} for ch in chains}
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"): continue
            if line[12:16].strip() != "CA": continue
            if line[16] not in (" ", "A"): continue
            ch = line[21]
            if ch not in chains: continue
            resnum = int(line[22:26])
            chain_coords[ch][resnum] = np.array([float(line[30:38]),
                                                  float(line[38:46]),
                                                  float(line[46:54])])
    common = set.intersection(*(set(chain_coords[ch]) for ch in chains))
    common_sorted = sorted(common)
    flat = np.concatenate([np.array([chain_coords[ch][r] for r in common_sorted]).flatten()
                           for ch in chains])
    residue_order = [(ch, i + 1) for ch in chains for i in range(len(common_sorted))]
    return flat, residue_order


def residue_mask(residue_order: list[tuple[str, int]], res_min: int, res_max: int) -> np.ndarray:
    """Boolean mask over a flattened CA-coordinate vector (3 entries per residue),
    True for CA atoms whose residue number falls in [res_min, res_max]."""
    mask = np.zeros(len(residue_order) * 3, dtype=bool)
    for i, (_, r) in enumerate(residue_order):
        if res_min <= r <= res_max:
            mask[3*i:3*i+3] = True
    return mask


def native_common_residues(pdb_path: Path, chains: list[str]) -> list[int]:
    """Native (author) residue numbers common to all `chains` in a reference PDB,
    sorted ascending. Position i in this list is the residue number corresponding
    to the i-th CA atom in parse_pdb_ca's/parse_cif_ca's per-chain output."""
    chain_res: dict[str, set[int]] = {ch: set() for ch in chains}
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"): continue
            if line[12:16].strip() != "CA": continue
            if line[16] not in (" ", "A"): continue
            ch = line[21]
            if ch not in chains: continue
            chain_res[ch].add(int(line[22:26]))
    common = set.intersection(*(chain_res[ch] for ch in chains))
    return sorted(common)


def parse_mmcif_ca_native(cif_path: Path, chain: str) -> dict[int, np.ndarray]:
    """CA coords for one chain from a full RCSB-style mmCIF file, keyed by the
    author (auth_seq_id) residue number -- unlike parse_cif_ca (for AF3's minimal
    CIFs), which keys by label_seq_id since AF3 renumbers each chain from 1."""
    coords: dict[int, np.ndarray] = {}
    in_atom, col = False, {}
    with open(cif_path) as fh:
        for line in fh:
            s = line.strip()
            if s == "loop_": in_atom = False; col = {}; continue
            if s.startswith("_atom_site."):
                col[s.split(".")[-1]] = len(col); in_atom = True; continue
            if not in_atom: continue
            if s.startswith("#"): in_atom = False; continue
            if not s.startswith("ATOM"): continue
            p = s.split()
            if p[col["label_atom_id"]] != "CA": continue
            if p[col["label_alt_id"]] not in (".", "A"): continue
            if p[col["auth_asym_id"]] != chain: continue
            resnum = int(p[col["auth_seq_id"]])
            coords[resnum] = np.array([float(p[col["Cartn_x"]]),
                                       float(p[col["Cartn_y"]]),
                                       float(p[col["Cartn_z"]])])
    return coords


# ── CIF parser ────────────────────────────────────────────────────────────────
def parse_cif_ca(cif_path: Path, chains: list[str]) -> np.ndarray | None:
    chain_coords: dict[str, dict[int, np.ndarray]] = {}
    in_atom, col = False, {}
    with open(cif_path) as fh:
        for line in fh:
            s = line.strip()
            if s == "loop_": in_atom = False; col = {}; continue
            if s.startswith("_atom_site."):
                col[s.split(".")[-1]] = len(col); in_atom = True; continue
            if not in_atom: continue
            if s.startswith("#"): in_atom = False; continue
            if not s.startswith("ATOM"): continue
            p = s.split()
            if p[col["label_atom_id"]] != "CA": continue
            ch = p[col["auth_asym_id"]]
            if ch not in chains: continue
            seq_id = int(p[col["label_seq_id"]])
            chain_coords.setdefault(ch, {})[seq_id] = np.array(
                [float(p[col["Cartn_x"]]), float(p[col["Cartn_y"]]), float(p[col["Cartn_z"]])])
    for ch in chains:
        if ch not in chain_coords: return None
    return np.concatenate([np.array([chain_coords[ch][r]
                           for r in sorted(chain_coords[ch])]).flatten()
                           for ch in chains])


# ── Kabsch superposition ──────────────────────────────────────────────────────
def kabsch_fit(mobile: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fits mobile (Nx3) onto ref (Nx3); returns (R, mobile_centroid, ref_centroid)."""
    mob_c, ref_c = mobile.mean(0), ref.mean(0)
    H = (mobile - mob_c).T @ (ref - ref_c)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, mob_c, ref_c


def apply_fit(coords: np.ndarray, R: np.ndarray, mob_c: np.ndarray, ref_c: np.ndarray) -> np.ndarray:
    return (coords - mob_c) @ R.T + ref_c


def superimpose(mobile_flat: np.ndarray, ref_flat: np.ndarray) -> np.ndarray:
    n = len(mobile_flat) // 3
    mob = mobile_flat.reshape(n, 3)
    ref = ref_flat.reshape(n, 3)
    R, mob_c, ref_c = kabsch_fit(mob, ref)
    return apply_fit(mob, R, mob_c, ref_c).flatten()


# ── ranking helper ────────────────────────────────────────────────────────────
def ranked_dirs(cond_dir: Path, top_n: int | None) -> list[tuple[int, int, Path]]:
    df = pd.read_csv(cond_dir / "ranking_scores.csv")
    df = df.sort_values("ranking_score", ascending=False)
    if top_n: df = df.head(top_n)
    result = []
    for _, row in df.iterrows():
        seed, sample = int(row["seed"]), int(row["sample"])
        d = cond_dir / f"seed-{seed}_sample-{sample}"
        if d.exists():
            result.append((seed, sample, d))
    return result


# ── parallel map helper ───────────────────────────────────────────────────────
def parallel_map(worker, jobs: list, workers: int) -> list:
    """Map `worker` over `jobs`. CIF parsing here is pure-Python string
    parsing, so real speedup needs separate processes (the GIL serializes
    Python bytecode across threads) -- hence a process Pool, not a thread
    pool. Falls back to a plain sequential loop when workers <= 1."""
    if workers <= 1 or len(jobs) <= 1:
        return [worker(j) for j in jobs]
    chunksize = max(1, len(jobs) // (workers * 4))
    with Pool(workers) as pool:
        return pool.map(worker, jobs, chunksize=chunksize)


# ── per-structure workers (module-level so they're picklable for Pool) ───────
def _load_condition_structure(job):
    seed, sample, sd, chains, ref_flat = job
    flat = parse_cif_ca(sd / "model.cif", chains)
    if flat is None or len(flat) != len(ref_flat):
        return None
    vec = superimpose(flat, ref_flat)
    return {"seed": seed, "sample": sample, "cif_path": str(sd / "model.cif"), "vec": vec}


def _load_combined_structure(job):
    variant, state, seed, sample, sd, chains, ref_flat = job
    flat = parse_cif_ca(sd / "model.cif", chains)
    if flat is None or len(flat) != len(ref_flat):
        return None
    vec = superimpose(flat, ref_flat)
    return {"variant": variant, "state": state, "seed": seed, "sample": sample,
            "cif_path": str(sd / "model.cif"), "vec": vec}


def _score_active_inactive(job):
    (variant, state, seed, sample, sd, ch, n_ref, core_pos, disc_pos,
     active_core, active_disc, inactive_disc) = job
    flat = parse_cif_ca(sd / "model.cif", [ch])
    if flat is None or len(flat) // 3 != n_ref:
        return None
    model      = flat.reshape(-1, 3)
    model_core = model[core_pos]
    model_disc = model[disc_pos]

    R, mc, rc          = kabsch_fit(model_core, active_core)
    model_disc_aligned = apply_fit(model_disc, R, mc, rc)

    rmsd_active   = float(np.sqrt(np.mean(np.sum((model_disc_aligned - active_disc) ** 2, axis=1))))
    rmsd_inactive = float(np.sqrt(np.mean(np.sum((model_disc_aligned - inactive_disc) ** 2, axis=1))))
    call = "active" if rmsd_active < rmsd_inactive else "inactive"

    row = {
        "variant": variant, "state": state, "chain": ch,
        "seed": seed, "sample": sample,
        "rmsd_active": round(rmsd_active, 3),
        "rmsd_inactive": round(rmsd_inactive, 3),
        "call": call,
        "cif_path": str(sd / "model.cif"),
    }
    return row, model_disc_aligned.flatten()


def _align_to_active(job):
    variant, state, seed, sample, sd, ch, n_ref, core_pos, feature_pos, active_core = job
    flat = parse_cif_ca(sd / "model.cif", [ch])
    if flat is None or len(flat) // 3 != n_ref:
        return None
    model      = flat.reshape(-1, 3)
    model_core = model[core_pos]
    model_feat = model[feature_pos]

    R, mc, rc          = kabsch_fit(model_core, active_core)
    model_feat_aligned = apply_fit(model_feat, R, mc, rc)

    meta = {"variant": variant, "state": state, "chain": ch,
            "seed": seed, "sample": sample, "cif_path": str(sd / "model.cif")}
    return meta, model_feat_aligned.flatten()


# ── silhouette k search ───────────────────────────────────────────────────────
def best_k(X: np.ndarray, k_min: int, k_max: int) -> int:
    best_k_, best_score = k_min, -1.0
    for k in range(k_min, min(k_max + 1, len(X))):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
        score  = silhouette_score(X, labels)
        if score > best_score:
            best_score, best_k_ = score, k
    return best_k_


# ── DBSCAN with auto eps ──────────────────────────────────────────────────────
def run_dbscan(X: np.ndarray, eps: float | None, min_samples: int):
    """
    Run DBSCAN. If eps is None, estimate it from the k-nearest neighbour
    distance distribution (elbow at the 90th percentile of 5-NN distances).
    Returns (labels, estimated_eps). Noise points get label -1.
    """
    if eps is None:
        k = min(min_samples, len(X) - 1)
        nbrs = NearestNeighbors(n_neighbors=k).fit(X)
        dists, _ = nbrs.kneighbors(X)
        eps = float(np.percentile(dists[:, -1], 90))
        print(f"  Auto eps (90th pct of {k}-NN dist): {eps:.4f}")

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    print(f"  DBSCAN: eps={eps:.4f}  min_samples={min_samples}  "
          f"→ {n_clusters} clusters  {n_noise} noise points")
    return labels, eps


# ── process one condition ─────────────────────────────────────────────────────
def process_condition(label: str, cond_dir: Path, ref_pdb: Path,
                      chains: list[str], top_n: int | None,
                      k_min: int, k_max: int, out_dir: Path,
                      forced_k: int | None = None,
                      method: str = "kmeans",
                      eps: float | None = None,
                      min_samples: int = 50,
                      file_prefix: str = "",
                      residues: tuple[int, int] | None = None,
                      workers: int = 1):

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    ref_flat, ref_res_order = parse_pdb_ca(ref_pdb, chains)
    n_ref    = len(ref_flat) // 3

    # load and superpose all structures
    sample_dirs = ranked_dirs(cond_dir, top_n)
    jobs = [(seed, sample, sd, chains, ref_flat) for seed, sample, sd in sample_dirs]
    results = parallel_map(_load_condition_structure, jobs, workers)

    vecs, meta = [], []
    for r in results:
        if r is None:
            continue
        vecs.append(r.pop("vec"))
        meta.append(r)

    if not vecs:
        print(f"  [SKIP] no valid structures")
        return

    print(f"  Loaded {len(vecs)} structures ({n_ref} CA atoms each)")

    # PCA — superposition above is always whole-structure (correct global frame);
    # if `residues` is set, only that residue window's CA coords feed the PCA so
    # the analysis is scoped to conformational variation local to that region.
    X_full = np.array(vecs)
    if residues is not None:
        mask = residue_mask(ref_res_order, residues[0], residues[1])
        if not mask.any():
            print(f"  [SKIP] no CA atoms found in residues {residues[0]}-{residues[1]}")
            return
        X = X_full[:, mask]
        print(f"  Restricting PCA to residues {residues[0]}-{residues[1]} "
              f"({mask.sum() // 3} CA atoms)")
    else:
        X = X_full
    n_pcs = min(10, len(vecs) - 1)
    pca   = PCA(n_components=n_pcs)
    X_pca = pca.fit_transform(X)
    exp   = pca.explained_variance_ratio_ * 100
    print(f"  PC1: {exp[0]:.1f}%  PC2: {exp[1]:.1f}%  PC3: {exp[2]:.1f}%")

    # cluster
    if method == "dbscan":
        labels, eps = run_dbscan(X_pca[:, :2], eps, min_samples)
        cluster_ids = sorted(set(labels))
        # compute centroids manually (exclude noise=-1)
        centroids = np.array([X_pca[labels == c, :2].mean(0)
                               for c in cluster_ids if c != -1])
        valid_ids = [c for c in cluster_ids if c != -1]
    else:
        if forced_k is not None:
            k = forced_k
            print(f"  k={k} (forced)")
        else:
            k = best_k(X_pca[:, :2], k_min, k_max)
            print(f"  Best k (silhouette, k={k_min}–{k_max}): {k}")
        km        = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels    = km.fit_predict(X_pca[:, :2])
        centroids = km.cluster_centers_
        valid_ids = list(range(k))

    # output dir for this condition
    res_tag   = f"_res{residues[0]}_{residues[1]}" if residues else ""
    cond_tag  = label.lower().replace(" ", "_") + res_tag
    cond_out  = out_dir / cond_tag
    cond_out.mkdir(parents=True, exist_ok=True)

    # save full cluster assignment for every structure (not just the reps) so
    # downstream analyses (e.g. averaging confidence metrics per cluster) can
    # join back to each seed/sample's AF3 output directory
    pd.DataFrame({
        "seed":    [m["seed"] for m in meta],
        "sample":  [m["sample"] for m in meta],
        "cluster": labels,
        "PC1":     X_pca[:, 0],
        "PC2":     X_pca[:, 1],
    }).to_csv(cond_out / "cluster_assignments.csv", index=False)

    # save representative per cluster
    reps = []
    print(f"  Cluster breakdown:")
    for i, c in enumerate(valid_ids):
        idxs     = np.where(labels == c)[0]
        centroid = centroids[i]
        dists    = np.linalg.norm(X_pca[idxs, :2] - centroid, axis=1)
        rep_idx  = idxs[np.argmin(dists)]

        stem = f"{file_prefix}_" if file_prefix else ""
        src  = Path(meta[rep_idx]["cif_path"])
        dest = cond_out / f"{stem}cluster_{c}_rep.cif"
        shutil.copy(src, dest)

        reps.append({
            "cluster": c, "size": len(idxs),
            "seed": meta[rep_idx]["seed"], "sample": meta[rep_idx]["sample"],
            "PC1": X_pca[rep_idx, 0], "PC2": X_pca[rep_idx, 1],
            "cif_path": str(dest),
        })
        print(f"    Cluster {c}: n={len(idxs):>5}  rep → seed-{meta[rep_idx]['seed']}_sample-{meta[rep_idx]['sample']}")

    if method == "dbscan":
        n_noise = (labels == -1).sum()
        if n_noise:
            print(f"    Noise:     n={n_noise:>5}  (not sampled)")

    pd.DataFrame(reps).to_csv(cond_out / "cluster_reps.csv", index=False)

    # plot
    # fixed palette; cluster id c always maps to colors[c % len(colors)]
    colors  = ["tab:blue", "tab:orange", "tab:green", "tab:purple",
               "tab:red", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    fig, ax = plt.subplots(figsize=(14, 11))

    # noise points (DBSCAN only)
    if -1 in labels:
        noise_idxs = np.where(labels == -1)[0]
        ax.scatter(X_pca[noise_idxs, 0], X_pca[noise_idxs, 1],
                   color="lightgrey", s=10, alpha=0.3, linewidths=0, label="Noise")

    for c in valid_ids:
        idxs = np.where(labels == c)[0]
        ax.scatter(X_pca[idxs, 0], X_pca[idxs, 1],
                   color=colors[c % len(colors)], s=18, alpha=0.4, linewidths=0,
                   label=f"Cluster {c} (n={len(idxs)})")

    # centroids
    ax.scatter(centroids[:, 0], centroids[:, 1],
               c="black", marker="x", s=200, linewidths=2.5, zorder=5)

    # rep stars
    for r in reps:
        ax.scatter(r["PC1"], r["PC2"], color=colors[r["cluster"] % len(colors)],
                   marker="*", s=500, edgecolors="black", linewidths=0.8, zorder=6)
        ax.annotate(f"C{r['cluster']}", (r["PC1"], r["PC2"]),
                    fontsize=16, xytext=(6, 4), textcoords="offset points")

    method_str  = (f"DBSCAN eps={eps:.3f}" if method == "dbscan"
                  else (f"k={forced_k} (forced)" if forced_k else f"k={len(valid_ids)} (silhouette)"))
    title_label = label + (f" (residues {residues[0]}-{residues[1]})" if residues else "")
    ax.set_xlabel(f"PC1 ({exp[0]:.1f}%)", fontsize=22)
    ax.set_ylabel(f"PC2 ({exp[1]:.1f}%)", fontsize=22)
    ax.set_title(f"{title_label}\n{method_str}  ★=rep  ×=centroid", fontsize=22)
    ax.tick_params(axis="both", labelsize=18)
    ax.legend(fontsize=18, loc="best", framealpha=0.7)
    plt.tight_layout()
    stem = f"{file_prefix}_" if file_prefix else ""
    png_path = cond_out / f"{stem}pca_clusters.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot → {png_path}")


# ── process combined conditions (pooled PCA across variants) ────────────────
def process_combined(conditions: list[tuple], top_n: int | None,
                     k_min: int, k_max: int, out_dir: Path,
                     forced_k: int | None = None,
                     method: str = "kmeans",
                     eps: float | None = None,
                     min_samples: int = 50,
                     color_key: str = "variant",
                     marker_key: str = "state",
                     out_subdir: str = "combined",
                     residues: tuple[int, int] | None = None,
                     workers: int = 1):
    """
    Pool structures from several conditions into a single PCA space so
    conformations can be compared directly, instead of each condition
    getting its own PCA. `color_key`/`marker_key` (each "variant" or
    "state") control how points are grouped in the scatter plot; e.g.
    color_key="variant" to compare WT/H524L/Y537S, or color_key="state"
    to compare apo vs holo within one variant. `residues` (min, max), if
    given, restricts the PCA input to that residue window's CA coords
    while superposition still uses the whole structure.
    """
    label_here = "Combined (" + ", ".join(sorted({c[0].split()[0] for c in conditions})) + ")"
    print(f"\n{'='*60}")
    print(f"  {label_here}")
    print(f"{'='*60}")

    ref_pdb, chains = conditions[0][2], conditions[0][3]
    ref_flat, ref_res_order = parse_pdb_ca(ref_pdb, chains)
    n_ref    = len(ref_flat) // 3

    jobs = []
    for label, cond_dir, cond_ref, cond_chains, file_prefix in conditions:
        if not cond_dir.exists():
            print(f"  [SKIP] {label}: {cond_dir} not found")
            continue
        variant, state = label.split()[0], label.split()[-1]
        for seed, sample, sd in ranked_dirs(cond_dir, top_n):
            jobs.append((variant, state, seed, sample, sd, cond_chains, ref_flat))

    results = parallel_map(_load_combined_structure, jobs, workers)

    vecs, meta = [], []
    for r in results:
        if r is None:
            continue
        vecs.append(r.pop("vec"))
        meta.append(r)

    if not vecs:
        print(f"  [SKIP] no valid structures")
        return

    print(f"  Loaded {len(vecs)} structures ({n_ref} CA atoms each) "
          f"from {len(conditions)} conditions")

    # PCA — superposition above is always whole-structure (correct global frame);
    # if `residues` is set, only that residue window's CA coords feed the PCA so
    # the analysis is scoped to conformational variation local to that region.
    X_full = np.array(vecs)
    if residues is not None:
        mask = residue_mask(ref_res_order, residues[0], residues[1])
        if not mask.any():
            print(f"  [SKIP] no CA atoms found in residues {residues[0]}-{residues[1]}")
            return
        X = X_full[:, mask]
        print(f"  Restricting PCA to residues {residues[0]}-{residues[1]} "
              f"({mask.sum() // 3} CA atoms)")
    else:
        X = X_full
    n_pcs = min(10, len(vecs) - 1)
    pca   = PCA(n_components=n_pcs)
    X_pca = pca.fit_transform(X)
    exp   = pca.explained_variance_ratio_ * 100
    print(f"  PC1: {exp[0]:.1f}%  PC2: {exp[1]:.1f}%  PC3: {exp[2]:.1f}%")

    # cluster (pooled)
    if method == "dbscan":
        labels, eps = run_dbscan(X_pca[:, :2], eps, min_samples)
        cluster_ids = sorted(set(labels))
        centroids = np.array([X_pca[labels == c, :2].mean(0)
                               for c in cluster_ids if c != -1])
        valid_ids = [c for c in cluster_ids if c != -1]
    else:
        if forced_k is not None:
            k = forced_k
            print(f"  k={k} (forced)")
        else:
            k = best_k(X_pca[:, :2], k_min, k_max)
            print(f"  Best k (silhouette, k={k_min}–{k_max}): {k}")
        km        = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels    = km.fit_predict(X_pca[:, :2])
        centroids = km.cluster_centers_
        valid_ids = list(range(k))

    res_tag  = f"_res{residues[0]}_{residues[1]}" if residues else ""
    cond_out = out_dir / (out_subdir + res_tag)
    cond_out.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "variant": [m["variant"] for m in meta],
        "state":   [m["state"] for m in meta],
        "seed":    [m["seed"] for m in meta],
        "sample":  [m["sample"] for m in meta],
        "cluster": labels,
        "PC1":     X_pca[:, 0],
        "PC2":     X_pca[:, 1],
    }).to_csv(cond_out / "cluster_assignments.csv", index=False)

    # save representative per cluster
    reps = []
    print(f"  Cluster breakdown:")
    for i, c in enumerate(valid_ids):
        idxs     = np.where(labels == c)[0]
        centroid = centroids[i]
        dists    = np.linalg.norm(X_pca[idxs, :2] - centroid, axis=1)
        rep_idx  = idxs[np.argmin(dists)]

        m    = meta[rep_idx]
        src  = Path(m["cif_path"])
        dest = cond_out / f"{m['variant']}_{m['state']}_cluster_{c}_rep.cif"
        shutil.copy(src, dest)

        reps.append({
            "cluster": c, "size": len(idxs),
            "variant": m["variant"], "state": m["state"],
            "seed": m["seed"], "sample": m["sample"],
            "PC1": X_pca[rep_idx, 0], "PC2": X_pca[rep_idx, 1],
            "cif_path": str(dest),
        })
        print(f"    Cluster {c}: n={len(idxs):>5}  rep → "
              f"{m['variant']} {m['state']} seed-{m['seed']}_sample-{m['sample']}")

    if method == "dbscan":
        n_noise = (labels == -1).sum()
        if n_noise:
            print(f"    Noise:     n={n_noise:>5}  (not sampled)")

    pd.DataFrame(reps).to_csv(cond_out / "cluster_reps.csv", index=False)

    # cluster composition: how much of each cluster is apo vs holo (and, when
    # multiple variants are pooled, how much is each variant) — lets you tell
    # a real state-mixed cluster from one that's just state-pure with stragglers
    variants_present = sorted(set(m["variant"] for m in meta))
    states_present   = sorted(set(m["state"] for m in meta))
    comp_rows = []
    for c in valid_ids:
        idxs = np.where(labels == c)[0]
        size = len(idxs)
        row  = {"cluster": c, "size": size}
        for v in variants_present:
            n = sum(1 for i in idxs if meta[i]["variant"] == v)
            row[f"n_{v}"], row[f"pct_{v}"] = n, round(100 * n / size, 1) if size else 0.0
        for s in states_present:
            n = sum(1 for i in idxs if meta[i]["state"] == s)
            row[f"n_{s}"], row[f"pct_{s}"] = n, round(100 * n / size, 1) if size else 0.0
        comp_rows.append(row)
    pd.DataFrame(comp_rows).to_csv(cond_out / "cluster_composition.csv", index=False)

    print(f"  Cluster composition:")
    for row in comp_rows:
        state_str = "  ".join(f"{s}={row[f'pct_{s}']:.0f}%" for s in states_present)
        print(f"    Cluster {row['cluster']}: n={row['size']:>5}  {state_str}")

    # plot: color = color_key, marker = marker_key; clustering shown via centroid/rep overlay
    palette       = ["tab:blue", "tab:orange", "tab:green", "tab:purple",
                     "tab:red", "tab:brown", "tab:pink", "tab:gray"]
    marker_shapes = ["o", "^", "s", "D", "P", "X"]

    meta_color  = np.array([m[color_key] for m in meta])
    meta_marker = np.array([m[marker_key] for m in meta])
    color_values  = sorted(set(meta_color))
    marker_values = sorted(set(meta_marker))
    color_map  = {v: palette[i % len(palette)] for i, v in enumerate(color_values)}
    marker_map = {v: marker_shapes[i % len(marker_shapes)] for i, v in enumerate(marker_values)}

    fig, ax = plt.subplots(figsize=(14, 11))

    for cv in color_values:
        for mv in marker_values:
            idxs = np.where((meta_color == cv) & (meta_marker == mv))[0]
            if len(idxs) == 0: continue
            point_label = cv if len(marker_values) <= 1 else f"{cv} {mv}"
            ax.scatter(X_pca[idxs, 0], X_pca[idxs, 1],
                       color=color_map[cv], marker=marker_map[mv], s=70, alpha=0.55,
                       edgecolors="black", linewidths=0.4,
                       label=f"{point_label} (n={len(idxs)})")

    # noise points (DBSCAN only)
    if -1 in labels:
        noise_idxs = np.where(labels == -1)[0]
        ax.scatter(X_pca[noise_idxs, 0], X_pca[noise_idxs, 1],
                   facecolors="none", edgecolors="lightgrey", s=70, linewidths=1,
                   label="Noise", zorder=4)

    # centroids
    ax.scatter(centroids[:, 0], centroids[:, 1],
               c="black", marker="x", s=200, linewidths=2.5, zorder=5)

    # rep stars, annotated with cluster id
    for r in reps:
        ax.scatter(r["PC1"], r["PC2"], color="none",
                   marker="*", s=500, edgecolors="black", linewidths=1.2, zorder=6)
        ax.annotate(f"C{r['cluster']}", (r["PC1"], r["PC2"]),
                    fontsize=16, xytext=(6, 4), textcoords="offset points")

    method_str   = (f"DBSCAN eps={eps:.3f}" if method == "dbscan"
                  else (f"k={forced_k} (forced)" if forced_k else f"k={len(valid_ids)} (silhouette)"))
    grouping_str = f"color={color_key}" + (f"  marker={marker_key}" if len(marker_values) > 1 else "")
    title_label  = label_here + (f" (residues {residues[0]}-{residues[1]})" if residues else "")
    ax.set_xlabel(f"PC1 ({exp[0]:.1f}%)", fontsize=22)
    ax.set_ylabel(f"PC2 ({exp[1]:.1f}%)", fontsize=22)
    ax.set_title(f"{title_label}\n{grouping_str}\n"
                 f"{method_str}  ★=rep  ×=centroid", fontsize=20)
    ax.tick_params(axis="both", labelsize=18)
    ax.legend(fontsize=16, loc="best", framealpha=0.7, markerscale=1.8)
    plt.tight_layout()
    png_path = cond_out / "combined_pca_clusters.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot → {png_path}")


# ── active vs. inactive H12 classification ───────────────────────────────────
def find_discriminating_residues(active: dict[int, np.ndarray], inactive: dict[int, np.ndarray],
                                 n_top: int = 15):
    """
    Aligns the inactive reference onto the active reference in two passes
    (rough full-chain fit, then refit using only the residues NOT among the
    most-different -- i.e. the stable core) and returns the residues that move
    the most between the two conformations (expected to fall on H12, without
    needing to hardcode its boundary) plus both references' coords in that
    shared frame.
    """
    common = sorted(set(active) & set(inactive))
    act       = np.array([active[r] for r in common])
    inact_raw = np.array([inactive[r] for r in common])

    R1, mc1, rc1  = kabsch_fit(inact_raw, act)
    inact_aligned1 = apply_fit(inact_raw, R1, mc1, rc1)
    dist1          = np.linalg.norm(act - inact_aligned1, axis=1)

    order         = np.argsort(-dist1)
    disc_residues = sorted(common[i] for i in order[:n_top])
    core_residues = [r for r in common if r not in set(disc_residues)]
    core_idx      = [common.index(r) for r in core_residues]

    R2, mc2, rc2  = kabsch_fit(inact_raw[core_idx], act[core_idx])
    inact_aligned2 = apply_fit(inact_raw, R2, mc2, rc2)

    active_by_res   = {r: act[i]            for i, r in enumerate(common)}
    inactive_by_res = {r: inact_aligned2[i] for i, r in enumerate(common)}
    return disc_residues, core_residues, active_by_res, inactive_by_res


def classify_active_inactive(active_cif: Path, active_chain: str,
                             inactive_cif: Path, inactive_chain: str,
                             ref_pdb: Path, ref_chains: list[str],
                             out_dir: Path, top_n: int | None = None,
                             n_disc: int = 15, workers: int = 1):
    """
    Scores every AF3-predicted monomer (each chain of every WT/H524L/Y537S
    dimer structure) for how closely its H12 region matches the active
    (agonist + coactivator peptide, `active_cif`, e.g. 3ERD) vs. inactive
    (antagonist, `inactive_cif`, e.g. 3ERT) reference conformation. The
    discriminating region is found automatically (the residues that differ
    most between the two references after core alignment) rather than
    hardcoding H12's boundary. Both references are real RCSB mmCIF files.
    """
    active_native   = parse_mmcif_ca_native(active_cif, active_chain)
    inactive_native = parse_mmcif_ca_native(inactive_cif, inactive_chain)
    disc_res, core_res, active_by_res, inactive_by_res = find_discriminating_residues(
        active_native, inactive_native, n_top=n_disc)
    print(f"Discriminating (H12-like) residues, native numbering: {disc_res}")

    # position (per-chain CA index) -> native residue number; AF3 renumbers
    # each chain 1..N from the construct start, so this shared positional map
    # (from the same dimer reference used elsewhere) translates model chains
    # (whatever chain letter) into the active/inactive references' numbering.
    pos_to_native = native_common_residues(ref_pdb, ref_chains)
    native_to_pos = {r: i for i, r in enumerate(pos_to_native)}
    disc_pos = [native_to_pos[r] for r in disc_res if r in native_to_pos]
    core_pos = [native_to_pos[r] for r in core_res if r in native_to_pos]
    if not disc_pos:
        sys.exit("No discriminating residues fall within the AF3 model's residue range")

    active_core   = np.array([active_by_res[pos_to_native[i]] for i in core_pos])
    active_disc   = np.array([active_by_res[pos_to_native[i]] for i in disc_pos])
    inactive_core = np.array([inactive_by_res[pos_to_native[i]] for i in core_pos])
    inactive_disc = np.array([inactive_by_res[pos_to_native[i]] for i in disc_pos])
    n_ref         = len(pos_to_native)

    dimer_conditions = [c for c in CONDITIONS
                        if c[0].split()[0] in ("WT", "H524L", "Y537S") and "dimer" in c[0]]

    jobs = []
    for label, cond_dir, _, chains, _ in dimer_conditions:
        if not cond_dir.exists():
            print(f"[SKIP] {label}: {cond_dir} not found")
            continue
        variant, state = label.split()[0], label.split()[-1]
        print(f"{label}: scoring...")
        for seed, sample, sd in ranked_dirs(cond_dir, top_n):
            for ch in chains:
                jobs.append((variant, state, seed, sample, sd, ch, n_ref, core_pos, disc_pos,
                            active_core, active_disc, inactive_disc))

    results = parallel_map(_score_active_inactive, jobs, workers)

    rows, disc_vecs = [], []
    for r in results:
        if r is None:
            continue
        row, vec = r
        rows.append(row)
        disc_vecs.append(vec)

    if not rows:
        print("[SKIP] no structures scored")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)

    # PCA on the H12/discriminating region only (already in a shared aligned
    # frame from the per-structure core fit above) -- lets you see whether the
    # active/inactive RMSD call corresponds to visually distinct conformations,
    # and gives PC1/PC2 for picking representative structures.
    n_pcs   = min(10, len(disc_vecs) - 1)
    pca     = PCA(n_components=n_pcs)
    X_pca   = pca.fit_transform(np.array(disc_vecs))
    exp     = pca.explained_variance_ratio_ * 100
    df["PC1"], df["PC2"] = X_pca[:, 0], X_pca[:, 1]
    print(f"\nH12-region PCA: PC1 {exp[0]:.1f}%  PC2 {exp[1]:.1f}%")

    df.to_csv(out_dir / "active_inactive_scores.csv", index=False)

    print("\nCall counts by variant/state:")
    print(df.groupby(["variant", "state", "call"]).size().unstack(fill_value=0))

    # representative structures: the most confidently active/inactive example
    # per variant/state, analogous to picking the cluster-centroid-closest rep
    # elsewhere in this script
    reps_dir = out_dir / "representative_structures"
    reps_dir.mkdir(parents=True, exist_ok=True)
    rep_rows = []
    for (variant, state), sub in df.groupby(["variant", "state"]):
        for call, score_col in (("active", "rmsd_active"), ("inactive", "rmsd_inactive")):
            called = sub[sub["call"] == call]
            if called.empty:
                continue
            best = called.loc[called[score_col].idxmin()]
            dest = reps_dir / f"{variant}_{state}_{call}_chain{best['chain']}_seed{best['seed']}_sample{best['sample']}.cif"
            shutil.copy(best["cif_path"], dest)
            rep_rows.append({
                "variant": variant, "state": state, "call": call,
                "chain": best["chain"], "seed": best["seed"], "sample": best["sample"],
                "rmsd_active": best["rmsd_active"], "rmsd_inactive": best["rmsd_inactive"],
                "cif_path": str(dest),
            })
    pd.DataFrame(rep_rows).to_csv(out_dir / "representative_structures.csv", index=False)
    print(f"\nRepresentative structures → {reps_dir}")

    # color = variant, marker (shape) = apo/holo state
    variant_colors = {"WT": "tab:blue", "H524L": "tab:orange", "Y537S": "tab:green"}
    state_markers  = {"apo": "o", "holo": "^"}

    # scatter 1: RMSD to active vs RMSD to inactive reference
    fig, ax = plt.subplots(figsize=(10, 9))
    for (variant, state), sub in df.groupby(["variant", "state"]):
        ax.scatter(sub["rmsd_active"], sub["rmsd_inactive"], s=45, alpha=0.55,
                   color=variant_colors.get(variant, "gray"), marker=state_markers.get(state, "o"),
                   edgecolors="black", linewidths=0.4,
                   label=f"{variant} {state} (n={len(sub)})")
    lims = [0, max(df["rmsd_active"].max(), df["rmsd_inactive"].max()) * 1.05]
    ax.plot(lims, lims, "k--", linewidth=1, alpha=0.6)
    ax.text(lims[1]*0.02, lims[1]*0.95, "INACTIVE\n(above diagonal)", fontsize=12,
            color="dimgray", ha="left", va="top")
    ax.text(lims[1]*0.95, lims[1]*0.02, "ACTIVE\n(below diagonal)", fontsize=12,
            color="dimgray", ha="right", va="bottom")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("RMSD to active reference (Å)", fontsize=16)
    ax.set_ylabel("RMSD to inactive reference (Å)", fontsize=16)
    ax.set_title(f"H12 active/inactive classification\n"
                 f"(discriminating residues: {disc_res[0]}-{disc_res[-1]}, native numbering)", fontsize=16)
    ax.legend(fontsize=11, loc="center left", framealpha=0.7, markerscale=1.3)
    plt.tight_layout()
    png_path = out_dir / "active_inactive_scatter.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot → {png_path}")

    # scatter 2: H12-region PCA, same color=variant / marker=state scheme,
    # with the active/inactive clusters labeled directly since they're
    # spatially separated
    fig, ax = plt.subplots(figsize=(11, 9))
    for (variant, state), sub in df.groupby(["variant", "state"]):
        ax.scatter(sub["PC1"], sub["PC2"], s=45, alpha=0.55,
                   color=variant_colors.get(variant, "gray"), marker=state_markers.get(state, "o"),
                   edgecolors="black", linewidths=0.4,
                   label=f"{variant} {state} (n={len(sub)})")
    for call, sub in df.groupby("call"):
        ax.annotate(call.upper(), (sub["PC1"].mean(), sub["PC2"].mean()),
                    fontsize=14, color="black", ha="center", va="center",
                    xytext=(0, 22), textcoords="offset points",
                    fontweight="bold")
    ax.set_xlabel(f"PC1 ({exp[0]:.1f}%)", fontsize=16)
    ax.set_ylabel(f"PC2 ({exp[1]:.1f}%)", fontsize=16)
    ax.set_title("H12-region PCA\n(marker: ● apo  ▲ holo)", fontsize=16)
    ax.legend(fontsize=11, loc="best", framealpha=0.7, markerscale=1.3)
    plt.tight_layout()
    pca_png_path = out_dir / "active_inactive_pca.png"
    fig.savefig(pca_png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot → {pca_png_path}")


# ── cluster relative to the active reference only ────────────────────────────
def cluster_vs_active(active_cif: Path, active_chain: str,
                      inactive_cif: Path, inactive_chain: str,
                      ref_pdb: Path, ref_chains: list[str],
                      out_dir: Path, top_n: int | None = None,
                      n_disc: int = 15,
                      k_min: int = 2, k_max: int = 8, forced_k: int | None = None,
                      method: str = "kmeans",
                      eps: float | None = None, min_samples: int = 50,
                      whole_structure: bool = False, workers: int = 1):
    """
    Instead of scoring every structure against both an active and an inactive
    reference, align every monomer onto the active reference only (core-fit
    on everything EXCEPT the H12/discriminating region, so global positioning
    noise -- and the very H12 displacement we're trying to observe -- doesn't
    bias the alignment) and let unsupervised clustering (kmeans/dbscan, same
    as the rest of this script) show where everything actually falls -- no
    inactive reference needed for scoring. `inactive_cif` is only used once,
    geometrically, to auto-locate which residues are H12 (the residues that
    move most between active/inactive), exactly as in classify_active_inactive.

    By default the PCA/clustering feature vector is just the H12 region's
    aligned coords, and each chain of a dimer is scored independently as its
    own row (2 rows per structure) -- appropriate for H12 since one protomer
    can be active-like while the other is inactive-like within the same
    dimer. With `whole_structure=True`, the feature vector instead covers
    every alignable residue (core + H12) AND the two chains' aligned vectors
    are concatenated into a single row per dimer structure (chain="AB"), so
    the PCA/clustering scores the whole complex as one unit (1 row per
    structure, matching the structure count) rather than two independent
    monomers. The active/inactive reference projections are tiled to match
    (both chains assumed active, or both inactive, since only monomeric
    active/inactive reference structures exist). The core-fit alignment
    itself (per chain, onto the monomeric active reference) is unchanged
    either way.
    """
    active_native   = parse_mmcif_ca_native(active_cif, active_chain)
    inactive_native = parse_mmcif_ca_native(inactive_cif, inactive_chain)
    disc_res, core_res, active_by_res, inactive_by_res = find_discriminating_residues(
        active_native, inactive_native, n_top=n_disc)
    print(f"H12-like region (native numbering): {disc_res}")

    pos_to_native = native_common_residues(ref_pdb, ref_chains)
    native_to_pos = {r: i for i, r in enumerate(pos_to_native)}
    disc_pos = [native_to_pos[r] for r in disc_res if r in native_to_pos]
    core_pos = [native_to_pos[r] for r in core_res if r in native_to_pos]
    if not disc_pos:
        sys.exit("No discriminating residues fall within the AF3 model's residue range")

    # feature_pos = what feeds the PCA. disc_pos only (H12) by default; with
    # whole_structure=True, every alignable position (core ∪ H12) -- the
    # core-fit below (which always excludes H12) is unaffected either way.
    feature_pos = sorted(core_pos + disc_pos) if whole_structure else disc_pos
    region_label = "whole structure" if whole_structure else "H12"

    active_core = np.array([active_by_res[pos_to_native[i]] for i in core_pos])
    active_feat = np.array([active_by_res[pos_to_native[i]] for i in feature_pos])
    # inactive_by_res is already in the active reference's frame (find_discriminating_residues
    # core-aligned it there), so it needs no further fitting -- same as active_feat.
    inactive_feat = np.array([inactive_by_res[pos_to_native[i]] for i in feature_pos])
    n_ref = len(pos_to_native)

    dimer_conditions = [c for c in CONDITIONS
                        if c[0].split()[0] in ("WT", "H524L", "Y537S") and "dimer" in c[0]]

    jobs = []
    for label, cond_dir, _, chains, _ in dimer_conditions:
        if not cond_dir.exists():
            print(f"[SKIP] {label}: {cond_dir} not found")
            continue
        variant, state = label.split()[0], label.split()[-1]
        print(f"{label}: aligning to active reference...")
        for seed, sample, sd in ranked_dirs(cond_dir, top_n):
            for ch in chains:
                jobs.append((variant, state, seed, sample, sd, ch, n_ref, core_pos, feature_pos,
                            active_core))

    results = parallel_map(_align_to_active, jobs, workers)

    if whole_structure:
        # Score the whole dimer as one unit: concatenate both chains' aligned
        # feature vectors into a single per-structure row (chain="AB") instead
        # of treating chain A/B as independent points, so the row count
        # matches the structure count (not 2x it).
        grouped: dict[tuple, dict[str, np.ndarray]] = {}
        cif_by_key: dict[tuple, str] = {}
        for r in results:
            if r is None:
                continue
            m, vec = r
            key = (m["variant"], m["state"], m["seed"], m["sample"])
            grouped.setdefault(key, {})[m["chain"]] = vec
            cif_by_key[key] = m["cif_path"]

        meta, feat_vecs = [], []
        for key, chain_vecs in grouped.items():
            if len(chain_vecs) < 2:
                continue  # missing a chain for this structure -- don't half-score it
            variant, state, seed, sample = key
            combined = np.concatenate([chain_vecs[ch] for ch in sorted(chain_vecs)])
            meta.append({"variant": variant, "state": state, "seed": seed, "sample": sample,
                        "chain": "AB", "cif_path": cif_by_key[key]})
            feat_vecs.append(combined)
    else:
        meta, feat_vecs = [], []
        for r in results:
            if r is None:
                continue
            m, vec = r
            meta.append(m)
            feat_vecs.append(vec)

    if not meta:
        print("[SKIP] no structures loaded")
        return

    X     = np.array(feat_vecs)
    n_pcs = min(10, len(X) - 1)
    pca   = PCA(n_components=n_pcs)
    X_pca = pca.fit_transform(X)
    exp   = pca.explained_variance_ratio_ * 100
    print(f"\n{region_label} PCA (active-aligned): PC1 {exp[0]:.1f}%  PC2 {exp[1]:.1f}%")

    # project the active/inactive references' own aligned coords through the
    # same PCA, so we can point to exactly where each lands and which cluster
    # the active one is nearest to. In whole_structure mode, rows are both
    # chains concatenated, so tile the (monomeric) reference to match --
    # representing "both protomers active" / "both protomers inactive".
    if whole_structure:
        active_feat_pca   = np.concatenate([active_feat, active_feat])
        inactive_feat_pca = np.concatenate([inactive_feat, inactive_feat])
    else:
        active_feat_pca, inactive_feat_pca = active_feat, inactive_feat
    active_ref_pc   = pca.transform(active_feat_pca.flatten().reshape(1, -1))[0, :2]
    inactive_ref_pc = pca.transform(inactive_feat_pca.flatten().reshape(1, -1))[0, :2]

    if method == "dbscan":
        labels, eps = run_dbscan(X_pca[:, :2], eps, min_samples)
        cluster_ids = sorted(set(labels))
        centroids = np.array([X_pca[labels == c, :2].mean(0) for c in cluster_ids if c != -1])
        valid_ids = [c for c in cluster_ids if c != -1]
    else:
        if forced_k is not None:
            k = forced_k
            print(f"k={k} (forced)")
        else:
            k = best_k(X_pca[:, :2], k_min, k_max)
            print(f"Best k (silhouette, k={k_min}-{k_max}): {k}")
        km        = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels    = km.fit_predict(X_pca[:, :2])
        centroids = km.cluster_centers_
        valid_ids = list(range(k))

    # which cluster is closest to the active reference itself
    active_ref_cluster = valid_ids[int(np.argmin(np.linalg.norm(centroids - active_ref_pc, axis=1)))]
    print(f"\nCluster {active_ref_cluster} is closest to the active reference "
          f"(distance {np.min(np.linalg.norm(centroids - active_ref_pc, axis=1)):.2f} in PC space)")

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(meta)
    df["cluster"], df["PC1"], df["PC2"] = labels, X_pca[:, 0], X_pca[:, 1]
    df.to_csv(out_dir / "cluster_vs_active_assignments.csv", index=False)

    # cluster composition: counts and % of each variant/state's structures
    # per cluster (rows sum to 100%, so you can see e.g. "1.6% of H524L apo
    # structures are in cluster 1" alongside the raw count)
    counts = pd.crosstab([df["variant"], df["state"]], df["cluster"])
    pcts   = pd.crosstab([df["variant"], df["state"]], df["cluster"], normalize="index") * 100
    counts.columns = [f"C{c}" if c != -1 else "noise" for c in counts.columns]
    pcts.columns   = counts.columns
    long_rows = []
    for (variant, state) in counts.index:
        for col in counts.columns:
            long_rows.append({"variant": variant, "state": state, "cluster": col,
                              "count": int(counts.loc[(variant, state), col]),
                              "pct": round(float(pcts.loc[(variant, state), col]), 1)})
    pd.DataFrame(long_rows).to_csv(out_dir / "cluster_composition.csv", index=False)

    display = counts.astype(str) + " (" + pcts.round(1).astype(str) + "%)"
    print("\nCluster composition (count (% of that variant/state's structures)):")
    print(display.to_string())

    # render the same table as a PNG
    row_labels = [f"{variant} {state}" for variant, state in display.index]
    col_labels = list(display.columns)
    active_col = f"C{active_ref_cluster}"
    fig_h = 0.6 + 0.4 * len(row_labels)
    fig, ax = plt.subplots(figsize=(1.8 * len(col_labels) + 2, fig_h))
    ax.axis("off")
    tbl = ax.table(cellText=display.values, rowLabels=row_labels, colLabels=col_labels,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1, 1.8)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0 and c >= 0 and col_labels[c] == active_col:
            cell.set_facecolor("#fff2b0")
            cell.set_text_props(fontweight="bold")
        if r == 0:
            cell.set_text_props(fontweight="bold")
    ax.set_title("Cluster composition (count (% of that variant/state's structures))\n"
                 f"{active_col} = closest to active reference", fontsize=13, pad=20)
    plt.tight_layout()
    comp_png_path = out_dir / "cluster_composition.png"
    fig.savefig(comp_png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot → {comp_png_path}")

    # representative structure per (cluster, variant, state): the structure of
    # that variant+state closest to the cluster's centroid, searched across
    # ALL structures of that variant+state (not just ones DBSCAN/kmeans
    # actually assigned to this cluster) -- so e.g. cluster 1 gets its own
    # nearest H524L-apo example even if the single globally-closest apo point
    # happens to be a different variant. `in_cluster` flags whether the pick
    # is an actual cluster member or just the closest thing around.
    reps_dir = out_dir / "representative_structures"
    reps_dir.mkdir(parents=True, exist_ok=True)
    all_states   = sorted(df["state"].unique())
    all_variants = sorted(df["variant"].unique())
    rep_rows = []
    for i, c in enumerate(valid_ids):
        for variant in all_variants:
            for state in all_states:
                sub = df[(df["variant"] == variant) & (df["state"] == state)]
                if sub.empty:
                    continue
                d          = np.linalg.norm(sub[["PC1", "PC2"]].values - centroids[i], axis=1)
                best_idx   = np.argmin(d)
                best       = sub.iloc[best_idx]
                in_cluster = bool(best["cluster"] == c)
                if not in_cluster:
                    print(f"  [nearest, not a member] cluster {c} {variant} {state}: "
                          f"closest is in cluster {best['cluster']}, dist={d[best_idx]:.2f}")
                dest = reps_dir / f"cluster{c}_{variant}_{state}_chain{best['chain']}_seed{best['seed']}_sample{best['sample']}.cif"
                shutil.copy(best["cif_path"], dest)
                rep_rows.append({"cluster": c, "variant": variant, "state": state,
                                 "in_cluster": in_cluster, "dist_to_centroid": round(float(d[best_idx]), 3),
                                 "chain": best["chain"], "seed": best["seed"], "sample": best["sample"],
                                 "cif_path": str(dest)})
    pd.DataFrame(rep_rows).to_csv(out_dir / "representative_structures.csv", index=False)
    print(f"\nRepresentative structures → {reps_dir}")

    variant_colors = {"WT": "tab:blue", "H524L": "tab:orange", "Y537S": "tab:green"}
    state_markers  = {"apo": "o", "holo": "^"}
    fig, ax = plt.subplots(figsize=(11, 9))
    for (variant, state), sub in df.groupby(["variant", "state"]):
        ax.scatter(sub["PC1"], sub["PC2"], s=45, alpha=0.55,
                   color=variant_colors.get(variant, "gray"), marker=state_markers.get(state, "o"),
                   edgecolors="black", linewidths=0.4, label=f"{variant} {state} (n={len(sub)})")
    ax.scatter(centroids[:, 0], centroids[:, 1], c="black", marker="x", s=200, linewidths=2.5, zorder=5)
    for i, c in enumerate(valid_ids):
        label = f"C{c} (= active ref)" if c == active_ref_cluster else f"C{c}"
        ax.annotate(label, centroids[i], fontsize=13, fontweight="bold" if c == active_ref_cluster else "normal",
                    xytext=(8, 8), textcoords="offset points", zorder=7)
    for r in rep_rows:
        if r["cif_path"] is None:
            continue
        rsub = df[(df["cluster"] == r["cluster"]) & (df["seed"] == r["seed"]) & (df["sample"] == r["sample"])
                  & (df["chain"] == r["chain"])]
        ax.scatter(rsub["PC1"], rsub["PC2"], color="none", marker="*", s=500,
                   edgecolors="black", linewidths=1.2, zorder=6)

    # mark exactly where the active/inactive references' own conformations project to
    ax.scatter(*active_ref_pc, marker="D", s=250, color="gold", edgecolors="black",
               linewidths=1.5, zorder=8, label=f"active reference ({active_cif.stem.upper()})")
    ax.scatter(*inactive_ref_pc, marker="D", s=250, color="dimgray", edgecolors="black",
               linewidths=1.5, zorder=8, label=f"inactive reference ({inactive_cif.stem.upper()})")

    method_str = (f"DBSCAN eps={eps:.3f}" if method == "dbscan"
                  else (f"k={forced_k} (forced)" if forced_k else f"k={len(valid_ids)} (silhouette)"))
    ax.set_xlabel(f"PC1 ({exp[0]:.1f}%)", fontsize=16)
    ax.set_ylabel(f"PC2 ({exp[1]:.1f}%)", fontsize=16)
    unit_str = "1 pt/dimer, both chains" if whole_structure else "1 pt/chain"
    ax.set_title(f"{region_label.capitalize()}, aligned to active reference only ({unit_str})\n"
                 f"{method_str}  ★=rep  ×=centroid  ◆=active/inactive ref  "
                 f"(C{active_ref_cluster} closest to active ref)", fontsize=14)
    ax.legend(fontsize=11, loc="best", framealpha=0.7, markerscale=1.3)
    plt.tight_layout()
    fname = "cluster_vs_active_pca_whole.png" if whole_structure else "cluster_vs_active_pca.png"
    png_path = out_dir / fname
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot → {png_path}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method",      choices=["kmeans", "dbscan"], default="kmeans",
                        help="Clustering method (default: kmeans)")
    parser.add_argument("--k",          type=int, default=None,
                        help="Force this k for all conditions (kmeans only), skipping silhouette search")
    parser.add_argument("--k_range",    type=int, nargs=2, default=[2, 8],
                        metavar=("MIN", "MAX"),
                        help="k range for silhouette search (default: 2 8)")
    parser.add_argument("--eps",         type=float, default=None,
                        help="DBSCAN eps (default: auto from 90th pct of 5-NN distances)")
    parser.add_argument("--min_samples", type=int, default=50,
                        help="DBSCAN min_samples (default: 50)")
    parser.add_argument("--top_n",      type=int, default=None,
                        help="Use only top N structures per condition (default: all)")
    parser.add_argument("--out_dir",    type=Path, default=None,
                        help="Output directory (default: "
                             "../outputs/analysis/updated_conformations, or "
                             "../outputs/analysis/combined_conformations with --combined)")
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="Only run these conditions (default: all). "
                             "Example: --conditions 'WT dimer holo' 'H524L dimer holo'")
    parser.add_argument("--combined", action="store_true",
                        help="Instead of a separate PCA per condition, pool structures into "
                             "combined PCA space(s). See --combined_by for grouping.")
    parser.add_argument("--combined_by", choices=["variant", "state"], default="variant",
                        help="Only with --combined. 'variant': one PCA pooling all of "
                             "WT/H524L/Y537S dimer holo+apo, colored by variant, marker=state "
                             "(default). 'state': one PCA per variant pooling its holo+apo "
                             "structures, colored by state — use this to compare apo vs holo.")
    parser.add_argument("--residues", type=int, nargs=2, default=None,
                        metavar=("MIN", "MAX"),
                        help="Restrict the PCA to CA atoms in this residue range (e.g. "
                             "--residues 219 239), scoping the analysis to conformational "
                             "variation local to that region. Superposition still uses the "
                             "whole structure.")
    parser.add_argument("--classify_active", action="store_true",
                        help="Instead of PCA/clustering, score every predicted monomer "
                             "(each chain of WT/H524L/Y537S dimer structures) for how "
                             "active- vs inactive-like its H12 region is, by RMSD to "
                             "--active_ref/--inactive_ref.")
    parser.add_argument("--active_ref", type=Path, default=INPUTS / "3erd.cif",
                        help="Active (agonist + coactivator peptide) monomer reference mmCIF "
                             "(default: inputs/3erd.cif). 1A52 is NOT used here -- without a "
                             "bound coactivator peptide its H12 is unresolved/displaced, not "
                             "capping the pocket.")
    parser.add_argument("--active_chain", default="A",
                        help="Chain to use from --active_ref (default: A)")
    parser.add_argument("--inactive_ref", type=Path, default=INPUTS / "3ert.cif",
                        help="Inactive (antagonist) monomer reference mmCIF (default: "
                             "inputs/3ert.cif)")
    parser.add_argument("--inactive_chain", default="A",
                        help="Chain to use from --inactive_ref (default: A)")
    parser.add_argument("--n_disc", type=int, default=15,
                        help="Number of most-different residues between the active/inactive "
                             "references to treat as the H12/discriminating region (default: 15)")
    parser.add_argument("--cluster_active", action="store_true",
                        help="Instead of scoring against both active and inactive references, "
                             "align every monomer's H12 region onto the active reference only "
                             "and cluster (kmeans/dbscan) to see where everything falls. "
                             "--inactive_ref is still used once, geometrically, to auto-locate "
                             "H12 -- not for scoring.")
    parser.add_argument("--whole_structure", action="store_true",
                        help="Only with --cluster_active. Run PCA/clustering over every "
                             "alignable residue (the whole monomer) instead of just the H12 "
                             "region. The active-reference alignment itself is unchanged "
                             "(still a core-fit excluding H12).")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of worker processes for parsing/aligning CIFs "
                             "(default: 1, sequential). CIF parsing is CPU-bound pure "
                             "Python, so this uses a process pool, not threads.")
    args = parser.parse_args()

    k_min, k_max = args.k_range
    residues = tuple(args.residues) if args.residues else None

    if args.classify_active:
        out_dir = args.out_dir or (OUTPUTS / "analysis" / "active_inactive")
        classify_active_inactive(args.active_ref, args.active_chain,
                                 args.inactive_ref, args.inactive_chain,
                                 INPUTS / "1A52_clean_dimer.pdb", ["A", "B"],
                                 out_dir, top_n=args.top_n, n_disc=args.n_disc,
                                 workers=args.workers)
        print("\n\nDone. Results in:", out_dir)
        return

    if args.cluster_active:
        default_name = "cluster_vs_active_whole" if args.whole_structure else "cluster_vs_active"
        out_dir = args.out_dir or (OUTPUTS / "analysis" / default_name)
        cluster_vs_active(args.active_ref, args.active_chain,
                          args.inactive_ref, args.inactive_chain,
                          INPUTS / "1A52_clean_dimer.pdb", ["A", "B"],
                          out_dir, top_n=args.top_n, n_disc=args.n_disc,
                          k_min=k_min, k_max=k_max, forced_k=args.k,
                          method=args.method, eps=args.eps, min_samples=args.min_samples,
                          whole_structure=args.whole_structure, workers=args.workers)
        print("\n\nDone. Results in:", out_dir)
        return

    if args.combined:
        out_dir = args.out_dir or (OUTPUTS / "analysis" / "combined_conformations")
        if args.combined_by == "variant":
            combined_conditions = [c for c in CONDITIONS
                                   if c[0].split()[0] in ("WT", "H524L", "Y537S")
                                   and "dimer" in c[0]]
            process_combined(combined_conditions, args.top_n, k_min, k_max, out_dir,
                             forced_k=args.k, method=args.method,
                             eps=args.eps, min_samples=args.min_samples,
                             color_key="variant", marker_key="state", out_subdir="combined",
                             residues=residues, workers=args.workers)
        else:  # "state" -> apo vs holo, one PCA per variant
            for variant in ("WT", "H524L", "Y537S"):
                subset = [c for c in CONDITIONS
                         if c[0].split()[0] == variant and "dimer" in c[0]]
                if not subset:
                    continue
                process_combined(subset, args.top_n, k_min, k_max, out_dir,
                                 forced_k=args.k, method=args.method,
                                 eps=args.eps, min_samples=args.min_samples,
                                 color_key="state", marker_key="variant",
                                 out_subdir=f"combined_{variant.lower()}",
                                 residues=residues, workers=args.workers)
        print("\n\nDone. Results in:", out_dir)
        return

    out_dir = args.out_dir or (OUTPUTS / "analysis" / "updated_conformations")

    conditions = CONDITIONS
    if args.conditions:
        conditions = [c for c in CONDITIONS if c[0] in args.conditions]
        if not conditions:
            sys.exit(f"No conditions matched: {args.conditions}")

    for label, cond_dir, ref_pdb, chains, file_prefix in conditions:
        if not cond_dir.exists():
            print(f"\n[SKIP] {label}: {cond_dir} not found")
            continue
        process_condition(label, cond_dir, ref_pdb, chains,
                          args.top_n, k_min, k_max, out_dir,
                          forced_k=args.k, method=args.method,
                          eps=args.eps, min_samples=args.min_samples,
                          file_prefix=file_prefix, residues=residues,
                          workers=args.workers)

    print("\n\nDone. Results in:", out_dir)


if __name__ == "__main__":
    main()
