#!/usr/bin/env python3
"""
interface_analyzer.py

Scores every mutant PDB in four H524 DMS/1ERE output directories:

  mono_apo    outputs/dms_H524_apo_1ERE          chain A only, no ligand — no interface,
                                                   just the pose's total Gibbs free energy (dG_complex)
  dimer_apo   outputs/dms_H524_dimer_apo_1ERE     chains A/B protein dimer, no ligand (A_B interface)
  mono_holo   outputs/dms_H524_holo_1ERE         chain A protein vs B estradiol    (A_B interface)
  dimer_holo  outputs/dms_H524_dimer_holo_1ERE    chains AB protein vs CD estradiol (AB_CD interface)

dG_complex (overall pose energy) is computed for all four conditions.
dG_interface/dG_receptor/dG_ligand, plus four interface-quality metrics
(dSASA, sc, packstat, unsat_hbonds -- buried interface area, shape
complementarity, packing quality, and buried unsatisfied polar atoms) are
additionally computed for dimer_apo, mono_holo and dimer_holo, where a real
interface exists. Each has a matching ddG-style delta column (ddSASA, dsc,
dpackstat, dunsat_hbonds) relative to that condition's own WT baseline.

Writes one combined CSV across all four conditions, and a single figure
with one heatmap panel per condition (one row of cells per condition, one
column per mutant amino acid): mono_apo's panel shows ddG_complex, the
other three panels show ddG_interface, each taken from that mutant's
lowest-energy replicate (by raw dG) — relative to that condition's own WT
(H524H) replicates. Panels get independent color scales since interface
ddG and whole-complex ddG aren't on the same magnitude. A heavier
horizontal line marks the apo/holo boundary in each panel.

Usage:
    conda activate pyrosetta
    python3 interface_analyzer.py --out_csv .../dms_H524_interface_scores.csv \
                                   --out_fig .../dms_H524_interface_heatmap.png
    python3 interface_analyzer.py --plot_only --out_csv <existing csv> --out_fig <png>
"""
import argparse
import csv
import logging
import os
import re
import sys
from multiprocessing import Pool

import pyrosetta
from pyrosetta import pose_from_pdb, create_score_function
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from pyrosetta.rosetta.protocols.constraint_movers import ClearConstraintsMover

logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
EST_PARAMS = os.path.join(REPO, "inputs", "est_ligand", "EST.params")

# ── The 4 conditions ─────────────────────────────────────────────────────────
CONDITIONS = [
    {
        "name": "mono_apo",
        "pdb_dir": os.path.join(REPO, "outputs", "dms_H524_apo_1ERE"),
        "interface": None,  # single chain, no ligand -> whole-complex energy only
        "receptor_chains": None,
        "ligand_chains": None,
        "metric": "dG_complex",
    },
    {
        "name": "dimer_apo",
        "pdb_dir": os.path.join(REPO, "outputs", "dms_H524_dimer_apo_1ERE"),
        "interface": "A_B",
        "receptor_chains": "A",
        "ligand_chains": "B",
        "metric": "dG_interface",
    },
    {
        "name": "mono_holo",
        "pdb_dir": os.path.join(REPO, "outputs", "dms_H524_holo_1ERE"),
        "interface": "A_B",
        "receptor_chains": "A",
        "ligand_chains": "B",
        "metric": "dG_interface",
    },
    {
        "name": "dimer_holo",
        "pdb_dir": os.path.join(REPO, "outputs", "dms_H524_dimer_holo_1ERE"),
        "interface": "AB_CD",
        "receptor_chains": "AB",
        "ligand_chains": "CD",
        "metric": "dG_interface",
    },
]

MUT_RE = re.compile(r"H524(?P<aa>[A-Z])(?P<wt>_wt)?_rep(?P<rep>\d+)")
ALL_AA = list("ACDEFGHIKLMNPQRSTVWY")  # includes WT residue H for the heatmap axis


# ── Worker initializer ───────────────────────────────────────────────────────
def init_worker():
    pyrosetta.init(f"-mute all -extra_res_fa {EST_PARAMS}", silent=True)


def _append(base_pose, extra_pose):
    base_pose.append_pose_by_jump(extra_pose, base_pose.num_jump() + 1)
    return base_pose


# ── Core scoring routine ─────────────────────────────────────────────────────
# Interface-only quality metrics pulled from InterfaceAnalyzerMover in addition to
# dG_interface: buried interface surface area (dSASA), shape complementarity (sc),
# packing quality (packstat), and buried unsatisfied interface polar atoms
# (unsat_hbonds) -- all computed on the same relaxed pose, no extra modeling needed.
IFACE_METRICS = ("dSASA", "sc", "packstat", "unsat_hbonds")


def calculate_metrics(pdb_path, interface, receptor_chains, ligand_chains):
    """Returns a dict with dG_interface, dG_complex, dG_receptor, dG_ligand, and
    the IFACE_METRICS. All but dG_complex are None when `interface` is None
    (single-chain pose)."""
    pose = pose_from_pdb(pdb_path)
    ClearConstraintsMover().apply(pose)
    scorefxn = create_score_function("ref2015")

    empty_iface = {m: None for m in IFACE_METRICS}
    if interface is None:
        dG_complex = scorefxn(pose)
        return {"dG_interface": None, "dG_complex": dG_complex, "dG_receptor": None,
                "dG_ligand": None, **empty_iface}

    iam = InterfaceAnalyzerMover(interface)
    iam.set_scorefunction(scorefxn)
    iam.set_pack_separated(True)
    iam.set_pack_rounds(5)
    iam.set_compute_packstat(True)
    iam.set_compute_interface_sc(True)
    iam.set_compute_interface_delta_hbond_unsat(True)
    iam.apply(pose)

    dG_interface = iam.get_interface_dG()
    dG_complex = iam.get_complex_energy()

    receptor_pose = None
    ligand_pose = None
    for p in pose.split_by_chain():
        chain_id = p.pdb_info().chain(1)
        if chain_id in receptor_chains:
            receptor_pose = p if receptor_pose is None else _append(receptor_pose, p)
        elif chain_id in ligand_chains:
            ligand_pose = p if ligand_pose is None else _append(ligand_pose, p)

    dG_receptor = scorefxn(receptor_pose)
    dG_ligand = scorefxn(ligand_pose)

    return {
        "dG_interface": dG_interface, "dG_complex": dG_complex,
        "dG_receptor": dG_receptor, "dG_ligand": dG_ligand,
        "dSASA": iam.get_interface_delta_sasa(),
        "sc": iam.get_all_data().sc_value,
        "packstat": iam.get_interface_packstat(),
        "unsat_hbonds": iam.get_interface_delta_hbond_unsat(),
    }


# ── One job = one mutant PDB in one condition ────────────────────────────────
def analyze_one(job):
    condition, pdb_path, interface, receptor_chains, ligand_chains, mutation, mut_aa, rep, is_wt = job
    try:
        metrics = calculate_metrics(pdb_path, interface, receptor_chains, ligand_chains)
        return {
            "condition": condition, "mutation": mutation, "mut_aa": mut_aa,
            "rep": rep, "is_wt": is_wt, "pdb_path": pdb_path,
            **metrics,
        }
    except Exception as exc:
        logger.error(f"Skipping {pdb_path}: {exc}")
        return None


# ── Job collection ───────────────────────────────────────────────────────────
def collect_jobs(condition):
    jobs = []
    for fname in sorted(os.listdir(condition["pdb_dir"])):
        if not fname.endswith(".pdb"):
            continue
        m = MUT_RE.search(fname)
        if not m:
            logger.warning(f"Filename didn't match H524 pattern, skipping: {fname}")
            continue
        mut_aa = m.group("aa")
        is_wt = m.group("wt") is not None
        rep = int(m.group("rep"))
        jobs.append((
            condition["name"],
            os.path.join(condition["pdb_dir"], fname),
            condition["interface"], condition["receptor_chains"], condition["ligand_chains"],
            f"H524{mut_aa}", mut_aa, rep, is_wt,
        ))
    return jobs


# ── Scoring: PyRosetta pass over all 4 conditions -> combined CSV ───────────
def run_scoring(out_csv, workers):
    jobs = [j for cond in CONDITIONS for j in collect_jobs(cond)]
    logger.info(f"Queued {len(jobs)} jobs across {len(CONDITIONS)} conditions ({workers} workers)")

    with Pool(processes=workers, initializer=init_worker) as pool:
        results = [r for r in pool.map(analyze_one, jobs) if r is not None]

    # baseline dG per condition = mean over that condition's WT (H524H) reps
    def _mean(rows, key):
        vals = [r[key] for r in rows if r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    delta_keys = ("dG_interface", "dG_complex", "dG_receptor", "dG_ligand", *IFACE_METRICS)

    baselines = {}
    for cond in CONDITIONS:
        wt_rows = [r for r in results if r["condition"] == cond["name"] and r["is_wt"]]
        baselines[cond["name"]] = {
            key: _mean(wt_rows, key) for key in delta_keys
        } if wt_rows else None

    fieldnames = ["condition", "mutation", "mut_aa", "rep", "is_wt", "pdb_path",
                  "dG_interface", "ddG_interface", "dG_complex", "ddG_complex",
                  "dG_receptor", "ddG_receptor", "dG_ligand", "ddG_ligand",
                  "dSASA", "ddSASA", "sc", "dsc", "packstat", "dpackstat",
                  "unsat_hbonds", "dunsat_hbonds"]
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(results, key=lambda r: (r["condition"], r["mut_aa"], r["rep"])):
            base = baselines[r["condition"]]
            row = dict(r)
            for key in delta_keys:
                if r[key] is not None and base and base[key] is not None:
                    row[f"d{key}"] = r[key] - base[key]
                else:
                    row[f"d{key}"] = ""
            writer.writerow(row)
    logger.info(f"Scores written to {out_csv}")


# ── Heatmap: one figure, 2 panels (complex ddG, interface ddG) ──────────────
def _heatmap_panel(ax, fig, df, aas, rows, metric_col, panel_title):
    """rows: list of condition names to draw, one per row of this panel.

    Each cell takes the ddG of the replicate with the lowest raw dG
    (i.e. the lowest-energy structure) for that mutant, rather than
    averaging ddG across replicates.
    """
    import matplotlib
    import numpy as np

    raw_col = metric_col[1:]  # "ddG_complex" -> "dG_complex", "ddG_interface" -> "dG_interface"

    matrix = []
    for cond_name in rows:
        sub = df[df["condition"] == cond_name]
        vals = []
        for aa in aas:
            aa_rows = sub[sub["mut_aa"] == aa].dropna(subset=[raw_col, metric_col])
            if aa_rows.empty:
                vals.append(float("nan"))
            else:
                best = aa_rows.loc[aa_rows[raw_col].idxmin()]
                vals.append(best[metric_col])
        matrix.append(vals)

    matrix = np.array(matrix, dtype=float)

    # Robust color scale: a single extreme mutant (e.g. a proline) shouldn't wash
    # out the rest of the panel to white. Scale to the 90th percentile of |ddG|
    # and let more extreme cells clip to the colormap's saturated end instead;
    # the exact value is still printed in each cell's annotation.
    finite = matrix[~np.isnan(matrix)]
    if finite.size:
        vmax = np.percentile(np.abs(finite), 90)
        vmax = vmax if vmax > 1e-9 else np.max(np.abs(finite))
    else:
        vmax = 1.0
    vmax = vmax if vmax > 1e-9 else 1.0
    norm = matplotlib.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    clipped_hi = bool(np.any(finite > vmax)) if finite.size else False
    clipped_lo = bool(np.any(finite < -vmax)) if finite.size else False
    extend = "both" if clipped_hi and clipped_lo else "max" if clipped_hi else "min" if clipped_lo else "neither"

    im = ax.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto")

    ax.set_xticks(range(len(aas)))
    ax.set_xticklabels(aas)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_title(panel_title, fontsize=11, loc="left")

    ax.set_xticks(np.arange(-0.5, len(aas), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Mark the apo/holo boundary (e.g. dimer_apo vs mono_holo/dimer_holo) with a
    # heavier line so the ligand-free comparison row reads as its own group.
    holo_idx = next((i for i, name in enumerate(rows) if "holo" in name), None)
    if holo_idx is not None and holo_idx > 0:
        ax.axhline(holo_idx - 0.5, color="black", linewidth=1.5)

    for i in range(len(rows)):
        for j in range(len(aas)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=7, color="black" if abs(val) < 0.6 * vmax else "white")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, extend=extend)
    cbar.set_label("ΔΔG (REU)", fontsize=8)


def plot_heatmap(csv_path, out_fig):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        sys.exit("matplotlib/pandas not found. Install with: pip install matplotlib pandas")

    df = pd.read_csv(csv_path)
    df = df[~df["is_wt"]]  # WT (ddG=0 by construction) doesn't need its own column

    aas = [a for a in ALL_AA if a != "H"]  # H is WT, not a scanned mutation

    complex_rows = [c["name"] for c in CONDITIONS]  # dG_complex computed for all 4
    interface_rows = [c["name"] for c in CONDITIONS if c["interface"] is not None]  # dimer_apo, mono_holo, dimer_holo

    fig, (ax_complex, ax_interface) = plt.subplots(
        2, 1,
        figsize=(0.55 * len(aas) + 2, 1.4 * len(complex_rows) + 1.4 * len(interface_rows) + 1),
        gridspec_kw={"height_ratios": [len(complex_rows), len(interface_rows)]},
    )

    _heatmap_panel(ax_complex, fig, df, aas, complex_rows, "ddG_complex",
                    "Whole-complex ΔG (all conditions)")
    _heatmap_panel(ax_interface, fig, df, aas, interface_rows, "ddG_interface",
                    "Interface ΔΔG (dimer_apo, mono_holo, dimer_holo)")

    ax_complex.set_xticklabels([])
    ax_interface.set_xlabel("Mutant residue at H524")

    fig.suptitle("ERα H524 DMS — ΔΔG relative to WT (1ERE)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    logger.info(f"Heatmap written to {out_fig}")


# ── Logging ───────────────────────────────────────────────────────────────────
def configure_logging(level=logging.INFO):
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(processName)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(handler)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="ddG scan across the 4 H524 DMS/1ERE conditions (apo/dimer_apo/holo/dimer_holo)"
    )
    parser.add_argument("--out_csv", default=os.path.join(REPO, "outputs", "dms_H524_interface_scores.csv"))
    parser.add_argument("--out_fig", default=os.path.join(REPO, "outputs", "dms_H524_interface_heatmap.png"))
    parser.add_argument("--workers", type=int,
                         default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    parser.add_argument("--plot_only", action="store_true",
                         help="Skip PyRosetta scoring; only regenerate the heatmap from --out_csv")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    configure_logging(logging.DEBUG if args.debug else logging.INFO)

    if not args.plot_only:
        run_scoring(args.out_csv, args.workers)
    plot_heatmap(args.out_csv, args.out_fig)


if __name__ == "__main__":
    main()
