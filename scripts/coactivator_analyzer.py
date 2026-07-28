#!/usr/bin/env python3
"""
coactivator_analyzer.py

Tests whether H524L/Y537S change coactivator (AF-2 groove) binding, as opposed
to the receptor-receptor dimer interface or receptor-ligand interface already
scored by interface_analyzer.py / interface_analyzer_Y537.py.

The AF-2 groove (H3/H4/H5/H12) is a different surface from the dimer
interface (mainly H10/H11) -- neither existing script touches it. This one
grafts the GRIP1/SRC-2 NR-box-II coactivator peptide (LXXLL motif) from PDB
3ERD (Shiau et al. 1998 -- ERa LBD + diethylstilbestrol + coactivator peptide,
inputs/3erd.cif) onto our own relaxed WT/H524L/Y537S apo and holo monomer
structures, by:

  1. Superposing 3ERD chain A onto the target's chain A using an iteratively
     trimmed CA Kabsch fit (3 rounds, dropping the worst 20% of residues each
     round) -- this converges on the rigid, conformation-invariant core of the
     LBD and avoids letting flexible loops/H12 itself bias the alignment.
  2. Applying that same rotation+translation to 3ERD chain C (the 11-residue
     peptide, sequence HKILHRLLQDS) to place it in the target's frame.
  3. Appending the transformed peptide as a new chain "C" and locally
     relaxing (FastRelax, InterfaceDesign2019 script -- same relax script the
     rest of this repo's DMS pipeline uses) with everything frozen except the
     peptide and an 8 A shell around it, so side chains/backbone can adapt to
     the mutant's specific sequence without disturbing the pre-relaxed rest
     of the structure.
  4. Scoring the resulting receptor-peptide interface with
     InterfaceAnalyzerMover, the same metrics as the other two analyzers
     (dG_interface, dG_complex, dSASA, sc, packstat, unsat_hbonds).

Sanity check before trusting a graft: CA distance from the peptide to the
canonical ERa AF-2 charge-clamp residues (Lys362, Glu542) should land close to
~8/~6 A, matching their distances in 3ERD's own native frame.

Targets (WT + mutant, apo + holo, for both positions) are the lowest-dG_complex
replicate already produced by the DMS pipeline -- see TARGETS below. Multiple
relax replicates per target are generated here (--nreps) and the lowest-energy
one is kept, same convention as the rest of the repo.

Usage:
    conda activate pyrosetta
    python3 coactivator_analyzer.py --out_csv .../coactivator_scores.csv \
                                     --out_dir .../coactivator_structures
"""
import argparse
import csv
import logging
import os
import sys
from multiprocessing import Pool

import numpy as np
import pyrosetta
from pyrosetta import pose_from_pdb, pose_from_file, create_score_function
from pyrosetta.rosetta.core.id import AtomID
from pyrosetta.rosetta.core.pose import Pose, append_subpose_to_pose
from pyrosetta.rosetta.core.select.residue_selector import (
    ChainSelector, NeighborhoodResidueSelector, NotResidueSelector,
)
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.task.operation import (
    InitializeFromCommandline, IncludeCurrent, ExtraRotamersGeneric,
    OperateOnResidueSubset, PreventRepackingRLT, RestrictToRepackingRLT,
)
from pyrosetta.rosetta.protocols.task_operations import LimitAromaChi2Operation
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from pyrosetta.rosetta.protocols.constraint_movers import ClearConstraintsMover
from pyrosetta.rosetta.numeric import xyzVector_double_t as V3

logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
EST_PARAMS = os.path.join(REPO, "inputs", "EST.params")
DONOR_CIF = os.path.join(REPO, "inputs", "3erd.cif")

PEPTIDE_CHAIN = "C"
DONOR_RECEPTOR_CHAIN = "A"
DONOR_PEPTIDE_CHAIN = "C"
CHARGE_CLAMP_RESIDUES = (362, 542)  # canonical ERa AF-2 charge clamp: Lys362, Glu542
IFACE_METRICS = ("dSASA", "sc", "packstat", "unsat_hbonds")

# ── Targets: lowest-dG_complex replicate already produced by the DMS pipeline ─
TARGETS = [
    {"label": "H524", "mutation": "H524H_wt", "is_wt": True, "condition": "apo",
     "pdb_path": os.path.join(REPO, "outputs", "dms_H524_apo_1ERE", "1ERE_apo_H524H_wt_rep3.pdb")},
    {"label": "H524", "mutation": "H524H_wt", "is_wt": True, "condition": "holo",
     "pdb_path": os.path.join(REPO, "outputs", "dms_H524_holo_1ERE", "1ERE_H524H_wt_rep5.pdb")},
    {"label": "H524", "mutation": "H524L", "is_wt": False, "condition": "apo",
     "pdb_path": os.path.join(REPO, "outputs", "dms_H524_apo_1ERE", "1ERE_apo_H524L_rep2.pdb")},
    {"label": "H524", "mutation": "H524L", "is_wt": False, "condition": "holo",
     "pdb_path": os.path.join(REPO, "outputs", "dms_H524_holo_1ERE", "1ERE_H524L_rep4.pdb")},
    {"label": "Y537", "mutation": "Y537Y_wt", "is_wt": True, "condition": "apo",
     "pdb_path": os.path.join(REPO, "outputs", "dms_Y537_apo_1ERE", "1ERE_apo_Y537Y_wt_rep1.pdb")},
    {"label": "Y537", "mutation": "Y537Y_wt", "is_wt": True, "condition": "holo",
     "pdb_path": os.path.join(REPO, "outputs", "dms_Y537_holo_1ERE", "1ERE_Y537Y_wt_rep4.pdb")},
    {"label": "Y537", "mutation": "Y537S", "is_wt": False, "condition": "apo",
     "pdb_path": os.path.join(REPO, "outputs", "dms_Y537_apo_1ERE", "1ERE_apo_Y537S_rep1.pdb")},
    {"label": "Y537", "mutation": "Y537S", "is_wt": False, "condition": "holo",
     "pdb_path": os.path.join(REPO, "outputs", "dms_Y537_holo_1ERE", "1ERE_Y537S_rep1.pdb")},
]


# ── Worker initializer ───────────────────────────────────────────────────────
def init_worker():
    pyrosetta.init(f"-mute all -extra_res_fa {EST_PARAMS}", silent=True)


# ── Step 1+2: robust core superposition, donor chain A onto target chain A ──
def _chain_ca_map(pose, chain_id):
    m = {}
    pi = pose.pdb_info()
    for i in range(1, pose.total_residue() + 1):
        if pi.chain(i) == chain_id and pose.residue(i).is_protein() and pose.residue(i).has("CA"):
            m[pi.number(i)] = np.array(pose.residue(i).xyz("CA"))
    return m


def _kabsch(P, Q):
    """R, t such that R @ P + t ~= Q (P, Q are Nx3 arrays)."""
    Pc, Qc = P.mean(0), Q.mean(0)
    P0, Q0 = P - Pc, Q - Qc
    H = P0.T @ Q0
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = Qc - R @ Pc
    return R, t


def robust_superpose_transform(donor_map, target_map, iters=3, keep_frac=0.8):
    """Iteratively-trimmed Kabsch fit on the residues common to both chain-A CA
    maps: fit, drop the worst-fitting (1 - keep_frac) residues, refit. Converges
    on the conformation-invariant structural core (rejects flexible loops/H12
    instead of letting them bias the whole-chain alignment)."""
    resids = sorted(set(donor_map) & set(target_map))
    for _ in range(iters):
        P = np.array([donor_map[r] for r in resids])
        Q = np.array([target_map[r] for r in resids])
        R, t = _kabsch(P, Q)
        err = np.linalg.norm((R @ P.T).T + t - Q, axis=1)
        keep_n = max(20, int(len(resids) * keep_frac))
        resids = [resids[i] for i in np.argsort(err)[:keep_n]]
    P = np.array([donor_map[r] for r in resids])
    Q = np.array([target_map[r] for r in resids])
    R, t = _kabsch(P, Q)
    rmsd = float(np.sqrt(np.mean(np.sum(((R @ P.T).T + t - Q) ** 2, axis=1))))
    return R, t, rmsd, len(resids)


# ── Step 3: graft the transformed peptide onto the target pose ─────────────
def graft_peptide(donor, target, R, t):
    pi = donor.pdb_info()
    c_idxs = [i for i in range(1, donor.total_residue() + 1) if pi.chain(i) == DONOR_PEPTIDE_CHAIN]
    pep = Pose()
    append_subpose_to_pose(pep, donor, c_idxs[0], c_idxs[-1], True)

    for i in range(1, pep.total_residue() + 1):
        res = pep.residue(i)
        for j in range(1, res.natoms() + 1):
            xyz = np.array(res.xyz(j))
            new_xyz = R @ xyz + t
            pep.set_xyz(AtomID(j, i), V3(*new_xyz))

    grafted = target.clone()
    n_before = grafted.total_residue()
    grafted.append_pose_by_jump(pep, n_before)

    pdb_info = grafted.pdb_info()
    for i in range(n_before + 1, grafted.total_residue() + 1):
        pdb_info.chain(i, PEPTIDE_CHAIN)
    grafted.pdb_info(pdb_info)
    return grafted


def charge_clamp_check(pose, chain_id):
    """CA distance from the peptide to Lys362/Glu542 -- sanity check that the
    graft landed in the real AF-2 groove, not somewhere degenerate."""
    pi = pose.pdb_info()
    pep_ca = []
    clamp = {}
    for i in range(1, pose.total_residue() + 1):
        if pi.chain(i) == PEPTIDE_CHAIN and pose.residue(i).has("CA"):
            pep_ca.append(np.array(pose.residue(i).xyz("CA")))
        elif pi.chain(i) == chain_id and pi.number(i) in CHARGE_CLAMP_RESIDUES:
            clamp[pi.number(i)] = np.array(pose.residue(i).xyz("CA"))
    pep_ca = np.array(pep_ca)
    dists = {r: float(np.linalg.norm(pep_ca - xyz, axis=1).min()) for r, xyz in clamp.items()}
    return dists


# ── Step 4: local relax restricted to the peptide + its shell ──────────────
def relax_interface(pose, scorefxn, shell_dist=8.0):
    pep_sel = ChainSelector(PEPTIDE_CHAIN)
    mobile_sel = NeighborhoodResidueSelector(pep_sel, shell_dist, True)
    frozen_sel = NotResidueSelector(mobile_sel)

    tf = TaskFactory()
    tf.push_back(InitializeFromCommandline())
    tf.push_back(IncludeCurrent())
    tf.push_back(ExtraRotamersGeneric())
    tf.push_back(LimitAromaChi2Operation())
    tf.push_back(OperateOnResidueSubset(PreventRepackingRLT(), frozen_sel))
    tf.push_back(OperateOnResidueSubset(RestrictToRepackingRLT(), mobile_sel))

    mobile_vec = mobile_sel.apply(pose)
    pep_vec = pep_sel.apply(pose)
    mm = MoveMap()
    mm.set_bb(False)
    mm.set_chi(False)
    for i in range(1, pose.total_residue() + 1):
        if mobile_vec[i]:
            mm.set_chi(i, True)
        if pep_vec[i]:
            mm.set_bb(i, True)
    mm.set_jump(pose.num_jump(), True)  # let the newly grafted peptide's rigid-body jump settle

    fr = FastRelax(scorefxn, "InterfaceDesign2019")
    fr.set_task_factory(tf)
    fr.set_movemap(mm)
    fr.apply(pose)
    return pose


# ── Step 5: score the receptor-peptide interface ────────────────────────────
def analyze_coactivator_interface(pose, receptor_chains, scorefxn):
    interface = f"{receptor_chains}_{PEPTIDE_CHAIN}"
    iam = InterfaceAnalyzerMover(interface)
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


# ── One job = one (target, replicate) ───────────────────────────────────────
def process_one(job):
    target, rep, out_dir = job
    try:
        pyrosetta.init(f"-mute all -extra_res_fa {EST_PARAMS}", silent=True)
        scorefxn = create_score_function("ref2015")

        donor = pose_from_file(DONOR_CIF)
        tgt_pose = pose_from_pdb(target["pdb_path"])
        ClearConstraintsMover().apply(tgt_pose)

        donor_map = _chain_ca_map(donor, DONOR_RECEPTOR_CHAIN)
        target_map = _chain_ca_map(tgt_pose, "A")
        R, t, core_rmsd, n_core = robust_superpose_transform(donor_map, target_map)

        grafted = graft_peptide(donor, tgt_pose, R, t)
        clamp_dists = charge_clamp_check(grafted, "A")

        receptor_chains = "A" if target["condition"] == "apo" else "AB"
        relax_interface(grafted, scorefxn)
        metrics = analyze_coactivator_interface(grafted, receptor_chains, scorefxn)

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            out_pdb = os.path.join(out_dir, f"{target['label']}_{target['mutation']}_{target['condition']}_rep{rep}.pdb")
            grafted.dump_pdb(out_pdb)

        return {
            "label": target["label"], "mutation": target["mutation"], "is_wt": target["is_wt"],
            "condition": target["condition"], "rep": rep,
            "core_rmsd": core_rmsd, "n_core_resid": n_core,
            "clamp_dist_362": clamp_dists.get(362), "clamp_dist_542": clamp_dists.get(542),
            **metrics,
        }
    except Exception as exc:
        logger.error(f"Skipping {target['label']}/{target['mutation']}/{target['condition']} rep{rep}: {exc}")
        return None


def run_scoring(out_csv, out_dir, nreps, workers):
    jobs = [(target, rep, out_dir) for target in TARGETS for rep in range(1, nreps + 1)]
    logger.info(f"Queued {len(jobs)} jobs ({len(TARGETS)} targets x {nreps} reps, {workers} workers)")

    with Pool(processes=workers, initializer=init_worker) as pool:
        results = [r for r in pool.map(process_one, jobs) if r is not None]

    # keep the lowest-dG_complex replicate per (label, mutation, condition)
    best = {}
    for r in results:
        key = (r["label"], r["mutation"], r["condition"])
        if key not in best or r["dG_complex"] < best[key]["dG_complex"]:
            best[key] = r

    # ddG relative to that position's own WT, per condition
    baselines = {}
    for (label, mutation, condition), r in best.items():
        if r["is_wt"]:
            baselines[(label, condition)] = r

    fieldnames = ["label", "mutation", "is_wt", "condition", "rep", "core_rmsd", "n_core_resid",
                  "clamp_dist_362", "clamp_dist_542",
                  "dG_interface", "ddG_interface", "dG_complex", "ddG_complex",
                  "dSASA", "ddSASA", "sc", "dsc", "packstat", "dpackstat",
                  "unsat_hbonds", "dunsat_hbonds"]
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for (label, mutation, condition), r in sorted(best.items()):
            base = baselines.get((label, condition))
            row = dict(r)
            for key in ("dG_interface", "dG_complex", *IFACE_METRICS):
                row[f"d{key}"] = (r[key] - base[key]) if base and r[key] is not None and base[key] is not None else ""
            writer.writerow(row)
    logger.info(f"Scores written to {out_csv}")


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


def main():
    parser = argparse.ArgumentParser(
        description="Graft the 3ERD coactivator peptide onto WT/H524L/Y537S apo+holo structures and score the AF-2 interface"
    )
    parser.add_argument("--out_csv", default=os.path.join(REPO, "outputs", "coactivator_scores.csv"))
    parser.add_argument("--out_dir", default=os.path.join(REPO, "outputs", "coactivator_structures"),
                         help="Where to dump grafted+relaxed structures (set to '' to skip)")
    parser.add_argument("--nreps", type=int, default=3, help="Relax replicates per target (default: 3)")
    parser.add_argument("--workers", type=int,
                         default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    configure_logging(logging.DEBUG if args.debug else logging.INFO)
    run_scoring(args.out_csv, args.out_dir or None, args.nreps, args.workers)


if __name__ == "__main__":
    main()
