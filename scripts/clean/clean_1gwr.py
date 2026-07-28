#!/usr/bin/env python3
"""
clean_1gwr.py

Cleans 1gwr.cif (Shiau et al. 2002 -- ERalpha LBD homodimer, each protomer bound
to its own 17beta-estradiol and its own TIF2 NR-box3 coactivator peptide, PDB
1GWR) for ER DMS + real-coactivator runs -- no peptide grafting needed, this
structure already has the LXXLL peptide bound in the AF-2 groove. Residue
numbering matches 1A52/1ERE (H524=His, Y537=Tyr confirmed at the same
auth_seq_id positions on chain A).

Four modes (same semantics as clean_1ERE.py/clean_1A52.py), always keeping the
coactivator peptide since that's the point of this structure:

  default       holo monomer + coactivator  — chain A protein, chain B estradiol,
                                                chain C peptide
  --apo         apo monomer + coactivator   — chain A protein, chain C peptide (no EST)
  --dimer       apo dimer + coactivator     — chains A/B protein, chains E/F peptide (no EST)
  --dimer-holo  holo dimer + coactivator    — chains A/B protein, chains C/D estradiol,
                                                chains E/F peptide

Removes waters in dimer modes; keeps chain-A waters in monomer modes unless --no_waters.

Usage (from scripts/):
    python clean_1gwr.py                                                       # holo monomer + coactivator
    python clean_1gwr.py --apo        --out_pdb ../inputs/1GWR/1GWR_clean_apo.pdb
    python clean_1gwr.py --dimer      --out_pdb ../inputs/1GWR/1GWR_clean_dimer_apo.pdb
    python clean_1gwr.py --dimer-holo --out_pdb ../inputs/1GWR/1GWR_clean_dimer.pdb
"""

import argparse
import os

from Bio.PDB import Chain, MMCIFParser, Model, PDBIO, Structure


def clean_cif(in_cif: str, out_pdb: str, protein_chains=("A", "B"), peptide_chains=("C", "D"),
              keep_waters: bool = True, apo: bool = False, dimer: bool = False, dimer_holo: bool = False):
    prot1_id, prot2_id = protein_chains
    pep1_id, pep2_id = peptide_chains

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("1GWR", in_cif)
    model_in = next(structure.get_models())

    def split_protein_chain(chain_id):
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

    prot1, est1, water1 = split_protein_chain(prot1_id)
    pep1 = list(model_in[pep1_id])

    out_structure = Structure.Structure("out")
    out_model = Model.Model(0)
    out_structure.add(out_model)

    def new_chain(chain_id, residues):
        ch = Chain.Chain(chain_id)
        for res in residues:
            ch.add(res.copy())
        out_model.add(ch)

    n_prot1 = n_prot2 = n_est1 = n_est2 = n_pep1 = n_pep2 = n_water = 0

    if dimer or dimer_holo:
        prot2, est2, water2 = split_protein_chain(prot2_id)
        pep2 = list(model_in[pep2_id])

        new_chain("A", prot1)
        new_chain("B", prot2)
        n_prot1, n_prot2 = len(prot1), len(prot2)

        if dimer_holo:
            new_chain("C", est1)
            new_chain("D", est2)
            n_est1, n_est2 = len(est1), len(est2)

        new_chain("E", pep1)
        new_chain("F", pep2)
        n_pep1, n_pep2 = len(pep1), len(pep2)
    else:
        new_chain("A", prot1)
        n_prot1 = len(prot1)

        if keep_waters and water1:
            water_chain = out_model["A"]
            for res in water1:
                water_chain.add(res.copy())
            n_water = len(water1)

        if not apo and est1:
            new_chain("B", est1)
            n_est1 = len(est1)

        new_chain("C", pep1)
        n_pep1 = len(pep1)

    io = PDBIO()
    io.set_structure(out_structure)
    io.save(out_pdb)

    print(f"Wrote {out_pdb}")
    print(f"  chain A protein: {n_prot1} residues")
    if dimer or dimer_holo:
        print(f"  chain B protein: {n_prot2} residues")
    if dimer_holo:
        print(f"  EST  chain C (mon 1): {n_est1} residues")
        print(f"  EST  chain D (mon 2): {n_est2} residues")
    elif n_est1:
        print(f"  EST  chain B ligand:  {n_est1} residues")
    if dimer or dimer_holo:
        print(f"  TIF2 peptide chain E (mon 1): {n_pep1} residues")
        print(f"  TIF2 peptide chain F (mon 2): {n_pep2} residues")
    else:
        print(f"  TIF2 peptide chain C: {n_pep1} residues")
    if n_water:
        print(f"  HOH  chain A waters: {n_water} residues")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(here))

    parser = argparse.ArgumentParser(description="Clean 1gwr.cif for ER DMS + real coactivator peptide")
    parser.add_argument("--in_cif",         default=os.path.join(repo, "inputs", "1GWR", "1gwr.cif"))
    parser.add_argument("--out_pdb",        default=os.path.join(repo, "inputs", "1GWR", "1GWR_clean.pdb"))
    parser.add_argument("--protein_chains", nargs=2, default=["A", "B"],
                        help="Which biological dimer copy's protein chains to keep")
    parser.add_argument("--peptide_chains", nargs=2, default=["C", "D"],
                        help="Matching native coactivator peptide chains (1GWR: C pairs with A, D pairs with B)")
    parser.add_argument("--no_waters",  action="store_true",
                        help="Strip water molecules (monomer modes only)")
    parser.add_argument("--apo",        action="store_true",
                        help="Omit estradiol (apo monomer + coactivator)")
    parser.add_argument("--dimer",      action="store_true",
                        help="Keep both protein chains + both peptides (homodimer, apo)")
    parser.add_argument("--dimer-holo", action="store_true", dest="dimer_holo",
                        help="Homodimer with both estradiols and both peptides")
    args = parser.parse_args()

    clean_cif(args.in_cif, args.out_pdb, protein_chains=tuple(args.protein_chains),
              peptide_chains=tuple(args.peptide_chains), keep_waters=not args.no_waters,
              apo=args.apo, dimer=args.dimer, dimer_holo=args.dimer_holo)


if __name__ == "__main__":
    main()
