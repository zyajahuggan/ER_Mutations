#!/usr/bin/env python3
"""
dms_H524.py

Deep mutation scan at H524 of the holo ERα monomer (chain A protein, chain B
estradiol). InterfaceAnalyzerMover scores the A_B interface (protein vs ligand).

Input:  --wt_pdb      monomer holo PDB (chain A protein, chain B estradiol)
        --est_params   Rosetta params for estradiol
        --xml          RosettaScripts XML (FastRelax + InterfaceAnalyzerMover, A_B interface)

Output: <out_dir>/<prefix>_<tag>_rep<n>.pdb
        <out_dir>/dms_H524_scores.csv

Workflow:
  1. Workers init PyRosetta with EST params and load the WT monomer pose once.
  2. Each job: clone WT → mutate H524 on chain A → FastRelax
               → InterfaceAnalyzer (A_B) → dump PDB.
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
DMS_RESNUM = 524
WT_AA      = "H"
ALL_AA     = list("ACDEFGIKLMNPQRSTVWY")

# ── Per-worker globals ─────────────────────────────────────────────────────────
GLOBAL_SCOREFXN = None
GLOBAL_WT_POSE  = None
GLOBAL_FRELAX   = None
GLOBAL_IAM      = None


def init_worker(wt_pdb: str, est_params: str, xml_file: str):
    """Load PyRosetta, WT monomer pose, and XML movers once per worker process."""
    pyrosetta.init(f"-mute all -extra_res_fa {est_params}", silent=True)

    global GLOBAL_SCOREFXN, GLOBAL_WT_POSE, GLOBAL_FRELAX, GLOBAL_IAM
    GLOBAL_SCOREFXN = pyrosetta.rosetta.core.scoring.get_score_function(True)
    GLOBAL_WT_POSE  = pyrosetta.io.pose_from_pdb(wt_pdb)

    xmlobj        = XmlObjects.create_from_file(xml_file)
    GLOBAL_FRELAX = xmlobj.get_mover("FastRelax")
    GLOBAL_IAM    = xmlobj.get_mover("analyze_interface")

    logger.debug("Worker ready: monomer holo WT pose loaded, movers configured")


# ── Job runner ─────────────────────────────────────────────────────────────────
def run_dms_job(job: tuple) -> dict | None:
    """Mutate chain A, relax, score protein-ligand interface. Returns score dict or None."""
    mut_aa, out_pdb, rep = job
    try:
        pose = GLOBAL_WT_POSE.clone()

        if mut_aa != WT_AA:
            pose_idx = pose.pdb_info().pdb2pose("A", DMS_RESNUM)
            logger.debug(f"Mutating A:{WT_AA}{DMS_RESNUM} → {mut_aa} (Rosetta res {pose_idx})")
            mutate_residue(pose, pose_idx, mut_aa,
                           pack_radius=5.0,
                           pack_scorefxn=GLOBAL_SCOREFXN)

        logger.debug(f"FastRelax: {WT_AA}{DMS_RESNUM}{mut_aa} rep{rep}")
        GLOBAL_FRELAX.apply(pose)

        logger.debug(f"InterfaceAnalyzer: {WT_AA}{DMS_RESNUM}{mut_aa} rep{rep}")
        GLOBAL_IAM.apply(pose)

        pose.dump_pdb(out_pdb)

        scores = {
            "mutation":     f"{WT_AA}{DMS_RESNUM}{mut_aa}",
            "mut_aa":       mut_aa,
            "rep":          rep,
            "out_pdb":      out_pdb,
            "dG_separated": pose.scores.get("dG_separated", float("nan")),
            "dG_cross":     pose.scores.get("dG_cross",     float("nan")),
            "total_score":  pose.scores.get("total_score",  float("nan")),
            "packstat":     pose.scores.get("packstat",     float("nan")),
            "hbonds_int":   pose.scores.get("hbonds_int",   float("nan")),
        }
        logger.info(
            f"Done: {scores['mutation']} rep{rep}  "
            f"dG={scores['dG_separated']:.2f}  total={scores['total_score']:.2f}"
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
    repo = os.path.dirname(os.path.dirname(here))

    parser = argparse.ArgumentParser(
        description="Holo monomer DMS at H524 of ERα — A_B interface"
    )
    parser.add_argument("--wt_pdb",     default=os.path.join(repo, "inputs", "1A52", "1A52_clean.pdb"),
                        help="Holo monomer PDB (chain A protein, chain B estradiol)")
    parser.add_argument("--est_params", default=os.path.join(repo, "inputs", "est_ligand", "EST.params"),
                        help="Rosetta .params file for estradiol")
    parser.add_argument("--xml",        default=os.path.join(here, "dms_H524.xml"),
                        help="RosettaScripts XML (FastRelax + InterfaceAnalyzerMover A_B)")
    parser.add_argument("--out_dir",    default=os.path.join(repo, "outputs", "dms_H524_holo"),
                        help="Directory for output PDBs and scores CSV")
    parser.add_argument("--nstruct",    type=int, default=5,
                        help="Relax replicates per mutant (default: 5)")
    parser.add_argument("--skip_wt",   action="store_true",
                        help="Do not run WT (H524H) as baseline")
    parser.add_argument("--workers",    type=int,
                        default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)),
                        help="Parallel processes (default: SLURM_CPUS_PER_TASK or cpu_count)")
    parser.add_argument("--prefix",     default="1A52",
                        help="Output PDB filename prefix (default: 1A52)")
    parser.add_argument("--debug",      action="store_true")
    args = parser.parse_args()

    configure_logging(logging.DEBUG if args.debug else logging.INFO)
    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.est_params):
        logger.error(
            f"EST params file not found: {args.est_params}\n"
            "Generate it with gen_EST_params.sh before running the DMS."
        )
        sys.exit(1)

    aas_to_run = ([] if args.skip_wt else [WT_AA]) + ALL_AA
    jobs = []
    for aa in aas_to_run:
        tag = f"{WT_AA}{DMS_RESNUM}H_wt" if aa == WT_AA else f"{WT_AA}{DMS_RESNUM}{aa}"
        for rep in range(1, args.nstruct + 1):
            out_pdb = os.path.join(args.out_dir, f"{args.prefix}_{tag}_rep{rep}.pdb")
            if os.path.exists(out_pdb):
                logger.debug(f"Skipping (exists): {os.path.basename(out_pdb)}")
                continue
            jobs.append((aa, out_pdb, rep))

    logger.info(f"Queued {len(jobs)} jobs  ({args.workers} workers)")

    with Pool(processes=args.workers,
              initializer=init_worker,
              initargs=(args.wt_pdb, args.est_params, args.xml)) as pool:
        all_scores = pool.map(run_dms_job, jobs)

    results = [r for r in all_scores if r is not None]
    out_csv = os.path.join(args.out_dir, "dms_H524_scores.csv")
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
    logger.info("Monomer holo DMS complete")
