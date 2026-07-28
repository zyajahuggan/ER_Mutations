#!/usr/bin/env python3
"""
clean_1ERE.py

Cleans 1ERE.cif for ER DMS runs (analogous to clean_1A52.py, but 1ERE is mmCIF
with 3 copies of the ERalpha dimer in the asymmetric unit: chains A/B, C/D, E/F,
each protein chain paired with its own EST ligand under the same auth_asym_id).
This script keeps only one dimer copy (chains A and B by default) and drops the
other two. Residue numbering matches 1A52 (H524, Y537 confirmed at the same
positions), so downstream DMS scripts work unchanged, just pointed at new
--wt_pdb / --out_dir paths.

Four modes (same semantics as clean_1A52.py):

  default       holo monomer  — chain A protein + EST renamed to chain B (A_B interface)
  --apo         apo monomer   — chain A protein only
  --dimer       apo dimer     — chains A and B protein, no ligand
  --dimer-holo  holo dimer    — chains A and B protein + EST A->C, EST B->D (AB_CD interface)

Removes waters in dimer modes; keeps chain-A waters in monomer modes unless --no_waters.

Usage (from scripts/):
    python clean_1ERE.py                                                    # holo monomer
    python clean_1ERE.py --apo        --out_pdb ../inputs/1ERE_clean_apo.pdb
    python clean_1ERE.py --dimer      --out_pdb ../inputs/1ERE_clean_dimer_apo.pdb
    python clean_1ERE.py --dimer-holo --out_pdb ../inputs/1ERE_clean_dimer.pdb
"""

import argparse
import os

from Bio.PDB import Chain, MMCIFParser, Model, PDBIO, Structure


def clean_cif(in_cif: str, out_pdb: str, chains=("A", "B"), keep_waters: bool = True,
              apo: bool = False, dimer: bool = False, dimer_holo: bool = False):
    chain1_id, chain2_id = chains

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("1ERE", in_cif)
    model_in = next(structure.get_models())

    def split_residues(chain_id):
        protein, est, water = [], [], []
        for res in model_in[chain_id]:
            resname = res.get_resname()
            hetfield = res.id[0]
            if resname == "HOH":
                water.append(res)
            elif resname == "EST":
                est.append(res)
            elif hetfield == " ":
                protein.append(res)
        return protein, est, water

    prot1, est1, water1 = split_residues(chain1_id)
    prot2, est2, water2 = split_residues(chain2_id)

    out_structure = Structure.Structure("out")
    out_model = Model.Model(0)
    out_structure.add(out_model)

    def new_chain(chain_id, residues):
        ch = Chain.Chain(chain_id)
        for res in residues:
            res = res.copy()
            ch.add(res)
        out_model.add(ch)
        return ch

    n_prot1 = n_prot2 = n_est_c = n_est_d = n_est_b = n_water = 0

    if dimer or dimer_holo:
        new_chain("A", prot1)
        new_chain("B", prot2)
        n_prot1, n_prot2 = len(prot1), len(prot2)

        if dimer_holo:
            new_chain("C", est1)
            new_chain("D", est2)
            n_est_c, n_est_d = len(est1), len(est2)
    else:
        new_chain("A", prot1)
        n_prot1 = len(prot1)

        if keep_waters and water1:
            # Waters stay logically part of chain A (matches clean_1A52.py behavior)
            water_chain = out_model["A"]
            for res in water1:
                water_chain.add(res.copy())
            n_water = len(water1)

        if not apo and est1:
            new_chain("B", est1)
            n_est_b = len(est1)

    io = PDBIO()
    io.set_structure(out_structure)
    io.save(out_pdb)

    print(f"Wrote {out_pdb}")
    print(f"  chain A protein ({chain1_id}):  {n_prot1} residues")
    if dimer or dimer_holo:
        print(f"  chain B protein ({chain2_id}):  {n_prot2} residues")
    if dimer_holo:
        print(f"  EST  chain C (mon 1):  {n_est_c} residues")
        print(f"  EST  chain D (mon 2):  {n_est_d} residues")
    elif n_est_b:
        print(f"  EST  chain B ligand:   {n_est_b} residues")
    if n_water:
        print(f"  HOH  chain A waters:   {n_water} residues")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)

    parser = argparse.ArgumentParser(description="Clean 1ERE.cif for ER DMS")
    parser.add_argument("--in_cif",      default=os.path.join(repo, "inputs", "1ERE.cif"))
    parser.add_argument("--out_pdb",     default=os.path.join(repo, "inputs", "1ERE_clean.pdb"))
    parser.add_argument("--chains",      nargs=2, default=["A", "B"],
                        help="Which biological dimer copy to keep, e.g. A B / C D / E F")
    parser.add_argument("--no_waters",   action="store_true",
                        help="Strip water molecules (monomer modes only)")
    parser.add_argument("--apo",         action="store_true",
                        help="Omit estradiol (apo monomer)")
    parser.add_argument("--dimer",       action="store_true",
                        help="Keep both protein chains (homodimer, apo)")
    parser.add_argument("--dimer-holo",  action="store_true", dest="dimer_holo",
                        help="Homodimer with both estradiol molecules (chains C and D)")
    args = parser.parse_args()

    clean_cif(args.in_cif, args.out_pdb, chains=tuple(args.chains),
              keep_waters=not args.no_waters, apo=args.apo,
              dimer=args.dimer, dimer_holo=args.dimer_holo)


if __name__ == "__main__":
    main()
