#!/usr/bin/env python3
"""
clean_1A52.py

Cleans 1A52.pdb for ER DMS runs. Four modes:

  default       holo monomer  — chain A protein + EST renamed to chain B (A_B interface)
  --apo         apo monomer   — chain A protein only
  --dimer       apo dimer     — chains A and B protein, no ligand
  --dimer-holo  holo dimer    — chains A and B protein + EST A→C, EST B→D (AB_CD interface)

Removes AU (gold) and waters in all modes.

Usage (from scripts/):
    python clean_1A52.py                                                   # holo monomer
    python clean_1A52.py --apo       --out_pdb ../inputs/1A52_clean_apo.pdb
    python clean_1A52.py --dimer     --out_pdb ../inputs/1A52_clean_dimer_apo.pdb
    python clean_1A52.py --dimer-holo --out_pdb ../inputs/1A52_clean_dimer.pdb
"""

import argparse
import os


def clean_pdb(in_pdb: str, out_pdb: str, keep_waters: bool = True, apo: bool = False,
              dimer: bool = False, dimer_holo: bool = False):
    chainA_lines = []
    chainB_lines = []
    water_lines  = []
    estC_lines   = []   # EST A renamed to chain C (monomer 1 ligand, dimer-holo)
    estD_lines   = []   # EST B renamed to chain D (monomer 2 ligand, dimer-holo)
    estB_lines   = []   # EST A renamed to chain B (monomer holo)

    with open(in_pdb) as fh:
        for raw in fh:
            rec = raw[:6].strip()

            if rec == "ATOM":
                if raw[21] == "A":
                    chainA_lines.append(raw)
                elif raw[21] == "B" and (dimer or dimer_holo):
                    chainB_lines.append(raw)

            elif rec == "HETATM":
                chain   = raw[21]
                resname = raw[17:20].strip()

                if resname == "AU":
                    continue

                if resname == "HOH":
                    if chain == "A" and keep_waters and not dimer and not dimer_holo:
                        water_lines.append(raw)

                elif resname == "EST":
                    if dimer_holo:
                        if chain == "A":
                            estC_lines.append(raw[:21] + "C" + raw[22:])
                        elif chain == "B":
                            estD_lines.append(raw[:21] + "D" + raw[22:])
                    elif not apo and not dimer and chain == "A":
                        # Monomer holo: rename EST A → chain B for A_B interface
                        estB_lines.append(raw[:21] + "B" + raw[22:])

    out_sections = []

    out_sections.extend(chainA_lines)
    out_sections.append("TER\n")

    if dimer or dimer_holo:
        out_sections.extend(chainB_lines)
        out_sections.append("TER\n")
    else:
        if water_lines:
            out_sections.extend(water_lines)
            out_sections.append("TER\n")

    if dimer_holo:
        out_sections.extend(estC_lines)
        out_sections.append("TER\n")
        out_sections.extend(estD_lines)
        out_sections.append("TER\n")
    elif estB_lines:
        out_sections.extend(estB_lines)
        out_sections.append("TER\n")

    out_sections.append("END\n")

    with open(out_pdb, "w") as fh:
        fh.writelines(out_sections)

    print(f"Wrote {out_pdb}")
    print(f"  ATOM chain A protein:  {len(chainA_lines)} records")
    if dimer or dimer_holo:
        print(f"  ATOM chain B protein:  {len(chainB_lines)} records")
    if dimer_holo:
        print(f"  EST  chain C (mon 1):  {len(estC_lines)} records")
        print(f"  EST  chain D (mon 2):  {len(estD_lines)} records")
    elif estB_lines:
        print(f"  EST  chain B ligand:   {len(estB_lines)} records")
    if water_lines:
        print(f"  HOH  chain A waters:   {len(water_lines)} records")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)

    parser = argparse.ArgumentParser(description="Clean 1A52 PDB for ER DMS")
    parser.add_argument("--in_pdb",      default=os.path.join(repo, "inputs", "1A52.pdb"))
    parser.add_argument("--out_pdb",     default=os.path.join(repo, "inputs", "1A52_clean.pdb"))
    parser.add_argument("--no_waters",   action="store_true",
                        help="Strip water molecules (monomer modes only)")
    parser.add_argument("--apo",         action="store_true",
                        help="Omit estradiol (apo monomer)")
    parser.add_argument("--dimer",       action="store_true",
                        help="Keep both protein chains A and B (homodimer, apo)")
    parser.add_argument("--dimer-holo",  action="store_true", dest="dimer_holo",
                        help="Homodimer with both estradiol molecules (chains C and D)")
    args = parser.parse_args()

    clean_pdb(args.in_pdb, args.out_pdb, keep_waters=not args.no_waters,
              apo=args.apo, dimer=args.dimer, dimer_holo=args.dimer_holo)


if __name__ == "__main__":
    main()
