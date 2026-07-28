#!/usr/bin/env python3
"""
rescore_coactivator_reps.py

coactivator_analyzer.py keeps only the lowest-dG_complex replicate per
(label, mutation, condition) in coactivator_scores.csv -- the other --nreps
relax replicates are discarded even though their PDBs are dumped to
outputs/coactivator_structures/. This rescores every dumped replicate so we
can see whether a given ddG signal (e.g. H524L holo) is consistent across
replicates or driven by a single lucky/unlucky relax.

Does not re-relax -- just reloads each already-relaxed dumped structure and
re-runs InterfaceAnalyzerMover on it, so scores are deterministic modulo
InterfaceAnalyzerMover's own repack-separated stochasticity.

Usage:
    conda activate pyrosetta
    python3 rescore_coactivator_reps.py --label H524
"""
import argparse
import glob
import os
import re

import pyrosetta
from pyrosetta import pose_from_pdb, create_score_function
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
STRUCT_DIR = os.path.join(REPO, "outputs", "coactivator_structures")
EST_PARAMS = os.path.join(REPO, "inputs", "EST.params")

TARGETS = {
    "H524": {"wt": "H524H_wt", "mut": "H524L"},
    "Y537": {"wt": "Y537Y_wt", "mut": "Y537S"},
}


def analyze(pose, receptor_chains, scorefxn):
    iam = InterfaceAnalyzerMover(f"{receptor_chains}_C")
    iam.set_scorefunction(scorefxn)
    iam.set_pack_separated(True)
    iam.set_pack_rounds(5)
    iam.set_compute_packstat(True)
    iam.set_compute_interface_sc(True)
    iam.set_compute_interface_delta_hbond_unsat(True)
    iam.apply(pose)
    return {
        "dG_interface": iam.get_interface_dG(),
        "dG_complex": iam.get_complex_energy(),
        "dSASA": iam.get_interface_delta_sasa(),
        "sc": iam.get_all_data().sc_value,
        "packstat": iam.get_interface_packstat(),
        "unsat_hbonds": iam.get_interface_delta_hbond_unsat(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, choices=TARGETS.keys())
    args = ap.parse_args()

    pyrosetta.init(f"-mute all -extra_res_fa {EST_PARAMS}", silent=True)
    scorefxn = create_score_function("ref2015")

    wt_mut = TARGETS[args.label]
    rows = []
    for condition in ("apo", "holo"):
        receptor_chains = "A" if condition == "apo" else "AB"
        for kind, mutation in (("wt", wt_mut["wt"]), ("mut", wt_mut["mut"])):
            pattern = os.path.join(STRUCT_DIR, f"{args.label}_{mutation}_{condition}_rep*.pdb")
            files = sorted(glob.glob(pattern), key=lambda p: int(re.search(r"rep(\d+)", p).group(1)))
            for f in files:
                rep = int(re.search(r"rep(\d+)", f).group(1))
                pose = pose_from_pdb(f)
                metrics = analyze(pose, receptor_chains, scorefxn)
                rows.append({"condition": condition, "mutation": mutation, "kind": kind, "rep": rep, **metrics})

    # print raw per-rep table
    print(f"\n{'condition':8} {'mutation':10} {'rep':4} {'dG_iface':>10} {'dG_cplx':>10} "
          f"{'sc':>6} {'packstat':>9} {'dSASA':>8} {'unsat':>6}")
    for r in rows:
        print(f"{r['condition']:8} {r['mutation']:10} {r['rep']:<4} "
              f"{r['dG_interface']:10.2f} {r['dG_complex']:10.2f} "
              f"{r['sc']:6.3f} {r['packstat']:9.3f} {r['dSASA']:8.1f} {r['unsat_hbonds']:6.0f}")

    # ddG_interface per mutant rep, relative to WT mean (same condition), plus
    # the mean/spread across mutant reps so a single-rep CSV pick can be judged
    print("\nddG_interface (mutant rep vs WT mean, same condition):")
    for condition in ("apo", "holo"):
        wt_vals = [r["dG_interface"] for r in rows if r["condition"] == condition and r["kind"] == "wt"]
        wt_mean = sum(wt_vals) / len(wt_vals)
        mut_vals = [r["dG_interface"] for r in rows if r["condition"] == condition and r["kind"] == "mut"]
        ddgs = [v - wt_mean for v in mut_vals]
        print(f"  {condition}: WT mean dG_interface = {wt_mean:.2f}  |  "
              f"mutant reps ddG = {[round(d, 2) for d in ddgs]}  |  "
              f"mean ddG = {sum(ddgs)/len(ddgs):.2f}")


if __name__ == "__main__":
    main()
