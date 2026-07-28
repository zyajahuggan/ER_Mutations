#!/usr/bin/env python3
"""
interface_analyzer_Y537_1GWR.py

Scores every mutant PDB in the four Y537 DMS/1GWR output directories:

  mono_apo    outputs/dms_Y537_apo_1GWR          chain A protein, chain C peptide, no ligand
                                                   (A_C peptide interface only -- lone monomer,
                                                    no dimer partner)
  dimer_apo   outputs/dms_Y537_dimer_apo_1GWR     chains A/B protein, chains E/F peptide, no ligand
                                                   (A_B protein-protein dimer interface +
                                                    AB_EF peptide interface)
  mono_holo   outputs/dms_Y537_holo_1GWR          chain A protein, chain B estradiol, chain C peptide
                                                   (A_B ligand interface + A_C peptide interface)
  dimer_holo  outputs/dms_Y537_dimer_holo_1GWR    chains A/B protein, chains C/D estradiol, chains E/F peptide
                                                   (AB_CD ligand interface + AB_EF peptide interface)

Unlike the plain 1ERE/1A52 DMS pipeline, the TIF2 coactivator peptide is present
in every 1GWR condition (that's the point of this structure), so a real
protein-peptide interface exists even in "apo" conditions. The "_lig"-labeled
columns follow the same convention interface_analyzer_Y537.py already uses for
its own dimer_apo condition: when there's no small-molecule ligand to occupy
that slot (dimer_apo), the A-B protein-protein dimerization interface fills it
instead, so dimer_apo still gets a "_lig" row in the same heatmap panel as the
true ligand interfaces (mono_holo, dimer_holo) -- it's just a different kind of
partner on the other side of the interface. mono_apo is the one condition with
no "_lig" interface at all, since it's a lone monomer with no dimer partner and
no ligand.

dG_complex (overall pose energy) is computed for all four conditions;
dG_interface_pep + four interface-quality metrics (dSASA_pep, sc_pep,
packstat_pep, unsat_hbonds_pep) are computed for all four (even the apo
conditions, since the peptide is always bound); the matching _lig columns are
computed for dimer_apo/mono_holo/dimer_holo and left blank only for mono_apo.
Each metric has a matching ddG-style delta column relative to that condition's
own WT baseline.

The DMS-time XML only ran a lean InterfaceAnalyzerMover (packstat only, no
sc/unsat_hbonds/dSASA), so this script re-scores each already-relaxed PDB with
those extra metrics turned on -- same reason interface_analyzer_Y537.py exists
for the 1ERE pipeline.

Writes one combined CSV across all four conditions, and a single figure with
three heatmap panels: whole-complex ddG (all 4 conditions), coactivator
peptide interface ddG (all 4 conditions, since the peptide is bound in both
apo and holo states), and interface ddG (dimer_apo, mono_holo, dimer_holo --
protein-protein dimer for dimer_apo, protein-ligand for the other two). Each
cell is that mutant's lowest-energy replicate (by raw dG), relative to that
condition's own WT (Y537Y) replicates.

Usage:
    conda activate pyrosetta
    python3 interface_analyzer_Y537_1GWR.py --out_csv .../dms_Y537_interface_scores_1GWR.csv \
                                             --out_fig .../dms_Y537_interface_heatmap_1GWR.png
    python3 interface_analyzer_Y537_1GWR.py --plot_only --out_csv <existing csv> --out_fig <png>
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
# lig_pair / pep_pair: (receptor_chains, partner_chains) for that interface, or
# None if that interface doesn't exist in this condition. The two interfaces
# use different chain groupings (e.g. dimer_apo's dimer interface is A_B, but
# its peptide interface is AB_EF), so each needs its own pair rather than a
# single shared receptor_chains.
CONDITIONS = [
    {
        "name": "mono_apo",
        "pdb_dir": os.path.join(REPO, "outputs", "dms_Y537_apo_1GWR"),
        "lig_pair": None,
        "pep_pair": ("A", "C"),
    },
    {
        "name": "dimer_apo",
        "pdb_dir": os.path.join(REPO, "outputs", "dms_Y537_dimer_apo_1GWR"),
        "lig_pair": ("A", "B"),      # no small-molecule ligand here -- the A-B protein-protein
                                      # dimer interface fills this slot instead, same convention
                                      # interface_analyzer_Y537.py uses for its own dimer_apo condition
        "pep_pair": ("AB", "EF"),
    },
    {
        "name": "mono_holo",
        "pdb_dir": os.path.join(REPO, "outputs", "dms_Y537_holo_1GWR"),
        "lig_pair": ("A", "B"),       # real ligand: estradiol
        "pep_pair": ("A", "C"),
    },
    {
        "name": "dimer_holo",
        "pdb_dir": os.path.join(REPO, "outputs", "dms_Y537_dimer_holo_1GWR"),
        "lig_pair": ("AB", "CD"),     # real ligand: both estradiols vs whole dimer
        "pep_pair": ("AB", "EF"),
    },
]

MUT_RE = re.compile(r"Y537(?P<aa>[A-Z])(?P<wt>_wt)?_rep(?P<rep>\d+)")
ALL_AA = list("ACDEFGHIKLMNPQRSTVWY")  # includes WT residue Y for the heatmap axis


# ── Worker initializer ───────────────────────────────────────────────────────
def init_worker():
    pyrosetta.init(f"-mute all -extra_res_fa {EST_PARAMS}", silent=True)


def _append(base_pose, extra_pose):
    base_pose.append_pose_by_jump(extra_pose, base_pose.num_jump() + 1)
    return base_pose


def _extract_chains(pose, chain_ids):
    sub = None
    for p in pose.split_by_chain():
        chain_id = p.pdb_info().chain(1)
        if chain_id in chain_ids:
            sub = p if sub is None else _append(sub, p)
    return sub


# ── Core scoring routine ─────────────────────────────────────────────────────
# Interface-only quality metrics pulled from InterfaceAnalyzerMover in addition to
# dG_interface: buried interface surface area (dSASA), shape complementarity (sc),
# packing quality (packstat), and buried unsatisfied interface polar atoms
# (unsat_hbonds) -- computed separately for whatever's in the "_lig" slot
# (protein-protein dimer for dimer_apo, protein-ligand for the holo conditions)
# and for the peptide interface (always present).
LIG_METRICS = ("dSASA_lig", "sc_lig", "packstat_lig", "unsat_hbonds_lig")
PEP_METRICS = ("dSASA_pep", "sc_pep", "packstat_pep", "unsat_hbonds_pep")


def _run_iam(pose, scorefxn, receptor_chains, partner_chains):
    iam = InterfaceAnalyzerMover(f"{receptor_chains}_{partner_chains}")
    iam.set_scorefunction(scorefxn)
    iam.set_pack_separated(True)
    iam.set_pack_rounds(5)
    iam.set_compute_packstat(True)
    iam.set_compute_interface_sc(True)
    iam.set_compute_interface_delta_hbond_unsat(True)
    iam.apply(pose)

    receptor_pose = _extract_chains(pose, receptor_chains)
    partner_pose = _extract_chains(pose, partner_chains)
    return {
        "dG_interface": iam.get_interface_dG(),
        "dG_receptor": scorefxn(receptor_pose),
        "dG_partner": scorefxn(partner_pose),
        "dSASA": iam.get_interface_delta_sasa(),
        "sc": iam.get_all_data().sc_value,
        "packstat": iam.get_interface_packstat(),
        "unsat_hbonds": iam.get_interface_delta_hbond_unsat(),
    }


def calculate_metrics(pdb_path, lig_pair, pep_pair):
    """Returns dG_complex plus _lig (only if lig_pair is set) and _pep interface metrics."""
    pose = pose_from_pdb(pdb_path)
    ClearConstraintsMover().apply(pose)
    scorefxn = create_score_function("ref2015")
    dG_complex = scorefxn(pose)

    lig_out = {"dG_interface_lig": None, "dG_receptor_lig": None, "dG_ligand": None, **{k: None for k in LIG_METRICS}}
    if lig_pair is not None:
        lig = _run_iam(pose, scorefxn, *lig_pair)
        lig_out = {
            "dG_interface_lig": lig["dG_interface"], "dG_receptor_lig": lig["dG_receptor"],
            "dG_ligand": lig["dG_partner"], "dSASA_lig": lig["dSASA"], "sc_lig": lig["sc"],
            "packstat_lig": lig["packstat"], "unsat_hbonds_lig": lig["unsat_hbonds"],
        }

    pep = _run_iam(pose, scorefxn, *pep_pair)
    pep_out = {
        "dG_interface_pep": pep["dG_interface"], "dG_receptor_pep": pep["dG_receptor"],
        "dG_peptide": pep["dG_partner"], "dSASA_pep": pep["dSASA"], "sc_pep": pep["sc"],
        "packstat_pep": pep["packstat"], "unsat_hbonds_pep": pep["unsat_hbonds"],
    }

    return {"dG_complex": dG_complex, **lig_out, **pep_out}


# ── One job = one mutant PDB in one condition ────────────────────────────────
def analyze_one(job):
    condition, pdb_path, lig_pair, pep_pair, mutation, mut_aa, rep, is_wt = job
    try:
        metrics = calculate_metrics(pdb_path, lig_pair, pep_pair)
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
            logger.warning(f"Filename didn't match Y537 pattern, skipping: {fname}")
            continue
        mut_aa = m.group("aa")
        is_wt = m.group("wt") is not None
        rep = int(m.group("rep"))
        jobs.append((
            condition["name"],
            os.path.join(condition["pdb_dir"], fname),
            condition["lig_pair"], condition["pep_pair"],
            f"Y537{mut_aa}", mut_aa, rep, is_wt,
        ))
    return jobs


# ── Scoring: PyRosetta pass over all 4 conditions -> combined CSV ───────────
def run_scoring(out_csv, workers):
    jobs = [j for cond in CONDITIONS for j in collect_jobs(cond)]
    logger.info(f"Queued {len(jobs)} jobs across {len(CONDITIONS)} conditions ({workers} workers)")

    with Pool(processes=workers, initializer=init_worker) as pool:
        results = [r for r in pool.map(analyze_one, jobs) if r is not None]

    def _mean(rows, key):
        vals = [r[key] for r in rows if r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    delta_keys = ("dG_complex",
                  "dG_interface_lig", "dG_receptor_lig", "dG_ligand", *LIG_METRICS,
                  "dG_interface_pep", "dG_receptor_pep", "dG_peptide", *PEP_METRICS)

    baselines = {}
    for cond in CONDITIONS:
        wt_rows = [r for r in results if r["condition"] == cond["name"] and r["is_wt"]]
        baselines[cond["name"]] = {key: _mean(wt_rows, key) for key in delta_keys} if wt_rows else None

    fieldnames = ["condition", "mutation", "mut_aa", "rep", "is_wt", "pdb_path",
                  "dG_complex", "ddG_complex",
                  "dG_interface_lig", "ddG_interface_lig", "dG_receptor_lig", "ddG_receptor_lig",
                  "dG_ligand", "ddG_ligand",
                  "dSASA_lig", "ddSASA_lig", "sc_lig", "dsc_lig", "packstat_lig", "dpackstat_lig",
                  "unsat_hbonds_lig", "dunsat_hbonds_lig",
                  "dG_interface_pep", "ddG_interface_pep", "dG_receptor_pep", "ddG_receptor_pep",
                  "dG_peptide", "ddG_peptide",
                  "dSASA_pep", "ddSASA_pep", "sc_pep", "dsc_pep", "packstat_pep", "dpackstat_pep",
                  "unsat_hbonds_pep", "dunsat_hbonds_pep"]
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


# ── Heatmap: one figure, 3 panels (complex ddG, peptide ddG, "_lig"-slot ddG) ─
def _heatmap_panel(ax, fig, df, aas, rows, metric_col, panel_title):
    """rows: list of condition names to draw, one per row of this panel.

    Each cell takes the ddG of the replicate with the lowest raw dG (i.e. the
    lowest-energy structure) for that mutant, rather than averaging ddG across
    replicates.
    """
    import matplotlib
    import numpy as np

    raw_col = metric_col[1:]  # "ddG_complex" -> "dG_complex", "ddG_interface_pep" -> "dG_interface_pep"

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

    aas = [a for a in ALL_AA if a != "Y"]  # Y is WT, not a scanned mutation

    complex_rows = [c["name"] for c in CONDITIONS]                          # all 4
    pep_rows = [c["name"] for c in CONDITIONS]                              # peptide always present, all 4
    lig_rows = [c["name"] for c in CONDITIONS if c["lig_pair"] is not None]  # dimer_apo, mono_holo, dimer_holo

    fig, (ax_complex, ax_pep, ax_lig) = plt.subplots(
        3, 1,
        figsize=(0.55 * len(aas) + 2, 1.4 * len(complex_rows) + 1.4 * len(pep_rows) + 1.4 * len(lig_rows) + 1.5),
        gridspec_kw={"height_ratios": [len(complex_rows), len(pep_rows), len(lig_rows)]},
    )

    _heatmap_panel(ax_complex, fig, df, aas, complex_rows, "ddG_complex",
                    "Whole-complex ΔG (all conditions)")
    _heatmap_panel(ax_pep, fig, df, aas, pep_rows, "ddG_interface_pep",
                    "Coactivator peptide interface ΔΔG (all conditions)")
    _heatmap_panel(ax_lig, fig, df, aas, lig_rows, "ddG_interface_lig",
                    "Interface ΔΔG (dimer_apo, mono_holo, dimer_holo)")

    ax_complex.set_xticklabels([])
    ax_pep.set_xticklabels([])
    ax_lig.set_xlabel("Mutant residue at Y537")

    fig.suptitle("ERα Y537 DMS — ΔΔG relative to WT (1GWR + TIF2 peptide)", y=1.02)
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
        description="ddG scan across the 4 Y537 DMS/1GWR conditions (apo/dimer_apo/holo/dimer_holo), "
                     "dimer/ligand and coactivator-peptide interfaces"
    )
    parser.add_argument("--out_csv", default=os.path.join(REPO, "outputs", "dms_Y537_interface_scores_1GWR.csv"))
    parser.add_argument("--out_fig", default=os.path.join(REPO, "outputs", "dms_Y537_interface_heatmap_1GWR.png"))
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
