#!/usr/bin/env python3
"""
dms_H524_dimer_apo_1GWR.py

Deep mutation scan at H524 of the apo ERα homodimer + real TIF2 coactivator peptides
(1GWR: chains A+B protein, chains E+F peptide, no estradiol). H524 is mutated
symmetrically on both chain A and chain B to model homozygous substitutions.
InterfaceAnalyzerMover scores the AB_EF protein-peptide interface.

Input:  inputs/1GWR_clean_dimer_apo.pdb       (chains A+B protein, E+F peptide, no ligand)
        scripts/dms_H524_dimer_apo_1GWR.xml   (FastRelax + InterfaceAnalyzerMover AB_EF)

Output: outputs/dms_H524_dimer_apo_1GWR/<tag>_rep<n>.pdb
        outputs/dms_H524_dimer_apo_1GWR/dms_H524_dimer_apo_scores.csv

Workflow:
  1. Workers init PyRosetta and load the WT homodimer pose once.
  2. Each job: clone WT → mutate H524 on both chain A and chain B → FastRelax
               → InterfaceAnalyzer (AB_EF, peptides) → dump PDB.
  3. WT (H524H) is included as the baseline.
  4. A summary CSV with dG_separated and total_score per mutant is written at the end.
"""

import csv
import logging
import os
import sys
import time
from multiprocessing import Pool

import pyrosetta
from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects
from pyrosetta.toolbox.mutants import mutate_residue

os.environ["TZ"] = "America/New_York"
time.tzset()
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
DMS_CHAINS = ["A", "B"]   # mutate H524 symmetrically on both monomers
DMS_RESNUM = 524
WT_AA      = "H"
ALL_AA     = list("ACDEFGIKLMNPQRSTVWY")

# ── Per-worker globals ─────────────────────────────────────────────────────────
GLOBAL_SCOREFXN = None
GLOBAL_WT_POSE  = None
GLOBAL_FRELAX   = None
GLOBAL_IAM_PEP  = None


def init_worker(wt_pdb: str, xml_file: str):
    """Load PyRosetta, WT homodimer pose, and XML movers once per worker process."""
    pyrosetta.init("-mute all", silent=True)

    global GLOBAL_SCOREFXN, GLOBAL_WT_POSE, GLOBAL_FRELAX, GLOBAL_IAM_PEP
    GLOBAL_SCOREFXN = pyrosetta.rosetta.core.scoring.get_score_function(True)
    GLOBAL_WT_POSE  = pyrosetta.io.pose_from_pdb(wt_pdb)

    xmlobj         = XmlObjects.create_from_file(xml_file)
    GLOBAL_FRELAX  = xmlobj.get_mover("FastRelax")
    GLOBAL_IAM_PEP = xmlobj.get_mover("analyze_interface_peptide")

    logger.debug("Worker ready: homodimer apo+peptide WT pose loaded, movers configured")


# ── Job runner ─────────────────────────────────────────────────────────────────
def run_dms_job(job: tuple) -> dict | None:
    """Mutate both chains, relax, score protein-peptide interface. Returns a score dict or None."""
    mut_aa, out_pdb, rep = job
    try:
        pose = GLOBAL_WT_POSE.clone()

        if mut_aa != WT_AA:
            for chain in DMS_CHAINS:
                pose_idx = pose.pdb_info().pdb2pose(chain, DMS_RESNUM)
                logger.debug(f"Mutating {chain}:{WT_AA}{DMS_RESNUM} → {mut_aa} (Rosetta res {pose_idx})")
                mutate_residue(pose, pose_idx, mut_aa,
                               pack_radius=5.0,
                               pack_scorefxn=GLOBAL_SCOREFXN)

        logger.debug(f"FastRelax: {WT_AA}{DMS_RESNUM}{mut_aa} rep{rep}")
        GLOBAL_FRELAX.apply(pose)

        logger.debug(f"InterfaceAnalyzer (peptides AB_EF): {WT_AA}{DMS_RESNUM}{mut_aa} rep{rep}")
        GLOBAL_IAM_PEP.apply(pose)

        pose.dump_pdb(out_pdb)

        scores = {
            "mutation":         f"{WT_AA}{DMS_RESNUM}{mut_aa}",
            "mut_aa":           mut_aa,
            "rep":              rep,
            "out_pdb":          out_pdb,
            "total_score":      pose.scores.get("total_score",  float("nan")),
            "dG_separated_pep": pose.scores.get("dG_separated", float("nan")),
            "dG_cross_pep":     pose.scores.get("dG_cross",     float("nan")),
            "packstat_pep":     pose.scores.get("packstat",     float("nan")),
            "hbonds_int_pep":   pose.scores.get("hbonds_int",   float("nan")),
        }
        logger.info(
            f"Done: {scores['mutation']} rep{rep}  "
            f"dG_pep={scores['dG_separated_pep']:.2f}  total={scores['total_score']:.2f}"
        )
        return scores

    except Exception as exc:
        logger.error(f"Failed {mut_aa} rep{rep}: {exc}", exc_info=True)
        return None


# ── Logging ────────────────────────────────────────────────────────────────────
def configure_logging(level=logging.INFO):
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(processName)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %Z",
    ))
    root.addHandler(handler)
    logging.captureWarnings(True)
    for noisy in ("pyrosetta", "jax"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    import argparse
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)

    parser = argparse.ArgumentParser(
        description="Apo homodimer DMS at H524 of ERα + TIF2 peptides (1GWR) — "
                     "symmetric mutation, AB_EF interface"
    )
    parser.add_argument("--wt_pdb",  default=os.path.join(repo, "inputs", "1GWR_clean_dimer_apo.pdb"),
                        help="Apo homodimer PDB (chains A+B protein, E+F peptide, no estradiol)")
    parser.add_argument("--xml",     default=os.path.join(here, "dms_H524_dimer_apo_1GWR.xml"),
                        help="RosettaScripts XML (FastRelax + InterfaceAnalyzerMover AB_EF)")
    parser.add_argument("--out_dir", default=os.path.join(repo, "outputs", "dms_H524_dimer_apo_1GWR"),
                        help="Directory for output PDBs and scores CSV")
    parser.add_argument("--nstruct", type=int, default=5,
                        help="Relax replicates per mutant (default: 5)")
    parser.add_argument("--skip_wt", action="store_true",
                        help="Do not run WT (H524H) as baseline")
    parser.add_argument("--workers", type=int,
                        default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)),
                        help="Parallel processes (default: SLURM_CPUS_PER_TASK or cpu_count)")
    parser.add_argument("--prefix",  default="1GWR",
                        help="Output PDB filename prefix (default: 1GWR)")
    parser.add_argument("--debug",   action="store_true")
    args = parser.parse_args()

    configure_logging(logging.DEBUG if args.debug else logging.INFO)
    os.makedirs(args.out_dir, exist_ok=True)

    aas_to_run = ([] if args.skip_wt else [WT_AA]) + ALL_AA
    jobs = []
    for aa in aas_to_run:
        tag = f"{WT_AA}{DMS_RESNUM}H_wt" if aa == WT_AA else f"{WT_AA}{DMS_RESNUM}{aa}"
        for rep in range(1, args.nstruct + 1):
            out_pdb = os.path.join(args.out_dir, f"{args.prefix}_dimer_apo_{tag}_rep{rep}.pdb")
            if os.path.exists(out_pdb):
                logger.debug(f"Skipping (exists): {os.path.basename(out_pdb)}")
                continue
            jobs.append((aa, out_pdb, rep))

    logger.info(f"Queued {len(jobs)} jobs  ({args.workers} workers)")

    with Pool(processes=args.workers,
              initializer=init_worker,
              initargs=(args.wt_pdb, args.xml)) as pool:
        all_scores = pool.map(run_dms_job, jobs)

    results = [r for r in all_scores if r is not None]
    out_csv = os.path.join(args.out_dir, "dms_H524_dimer_apo_scores.csv")
    if results:
        fieldnames = list(results[0].keys())
        with open(out_csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sorted(results, key=lambda r: (r["mut_aa"], r["rep"])))
        logger.info(f"Scores written to {out_csv}")
    else:
        logger.warning("No successful jobs — CSV not written")


if __name__ == "__main__":
    main()
    logger.info("Homodimer apo+peptide DMS complete")
