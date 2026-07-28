#!/usr/bin/env python3
"""
Average AF3 confidence metrics within each structural cluster.

Reads cluster_assignments.csv (written by af3_conformations.py, one row per
structure with its cluster label) for each condition, pulls per-structure
confidence values from that sample's summary_confidences.json (and
confidences.json for mean pLDDT, if --plddt is passed), and averages them
within each cluster.

Requires af3_conformations.py to have been (re)run with the
cluster_assignments.csv output first.

Usage (from the scripts/af3/ directory):
    python cluster_confidence.py [--conf_dir DIR] [--plddt] [--conditions ...]

    --conf_dir   : directory of per-condition subfolders holding
                   cluster_assignments.csv (default:
                   outputs/analysis/archive/per_condition_clusters_confidence_2026-07-06)
    --plddt      : also average mean per-structure pLDDT (slower - loads
                   confidences.json, which includes the full PAE matrix)
    --conditions : only run these conditions (default: all)
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from af3_conformations import CONDITIONS, OUTPUTS

METRICS = ["ranking_score", "iptm", "ptm", "fraction_disordered", "has_clash"]


def load_confidence(sample_dir: Path, want_plddt: bool) -> dict | None:
    sc_path = sample_dir / "summary_confidences.json"
    if not sc_path.exists():
        return None
    with open(sc_path) as fh:
        d = json.load(fh)
    row = {m: d.get(m) for m in METRICS}

    if want_plddt:
        c_path = sample_dir / "confidences.json"
        if c_path.exists():
            with open(c_path) as fh:
                c = json.load(fh)
            row["mean_plddt"] = float(np.mean(c["atom_plddts"]))
        else:
            row["mean_plddt"] = None

    return row


def process_condition(label: str, cond_dir: Path, conf_dir: Path, want_plddt: bool):
    cond_tag = label.lower().replace(" ", "_")
    cond_out = conf_dir / cond_tag
    assign_path = cond_out / "cluster_assignments.csv"

    if not assign_path.exists():
        print(f"[SKIP] {label}: no cluster_assignments.csv "
              f"(rerun af3_conformations.py to generate it)")
        return None

    assign = pd.read_csv(assign_path)

    rows = []
    for _, r in assign.iterrows():
        sd = cond_dir / f"seed-{int(r['seed'])}_sample-{int(r['sample'])}"
        conf = load_confidence(sd, want_plddt)
        if conf is None:
            continue
        conf["cluster"] = r["cluster"]
        rows.append(conf)

    if not rows:
        print(f"[SKIP] {label}: no confidence JSONs found")
        return None

    cdf = pd.DataFrame(rows)
    metrics = METRICS + (["mean_plddt"] if want_plddt else [])
    # keep noise (cluster == -1, DBSCAN only) out of the per-cluster averages
    cdf = cdf[cdf["cluster"] != -1]

    summary = cdf.groupby("cluster")[metrics].agg(["mean", "std", "count"])
    summary.to_csv(cond_out / "cluster_confidence_summary.csv")

    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    print(summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--conf_dir", type=Path,
                        default=OUTPUTS / "analysis" / "archive" / "per_condition_clusters_confidence_2026-07-06",
                        help="Directory containing per-condition cluster_assignments.csv files")
    parser.add_argument("--plddt", action="store_true",
                        help="Also average mean per-structure pLDDT (slower)")
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="Only run these conditions (default: all)")
    args = parser.parse_args()

    conditions = CONDITIONS
    if args.conditions:
        conditions = [c for c in CONDITIONS if c[0] in args.conditions]
        if not conditions:
            sys.exit(f"No conditions matched: {args.conditions}")

    for label, cond_dir, ref_pdb, chains, file_prefix in conditions:
        process_condition(label, cond_dir, args.conf_dir, args.plddt)


if __name__ == "__main__":
    main()
