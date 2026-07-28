#!/usr/bin/env python3
"""Merge batched AF3 outputs back into their canonical condition folder.

For each canonical folder (e.g. 1a52_wt_apo), finds sibling batch folders
named {condition}_s{start}_{end} (as produced by split_af3_seed_batches.py),
moves their seed-*_sample-* subdirectories into the canonical folder, and
appends their ranking_scores.csv rows onto the canonical ranking_scores.csv.
Downstream scripts (af3_pca.py, af3_conformations.py, af3_cluster_sample.py)
only read seed-*_sample-* dirs + ranking_scores.csv, so batch-level top-level
files (model.cif, confidences.json etc., which only reflect the best seed
within that batch) are discarded once merged.
"""
import argparse
import csv
import re
import shutil
from pathlib import Path

BATCH_RE = re.compile(r"^(?P<condition>.+)_s\d+_\d+$")


def find_batches(output_dir: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for entry in output_dir.iterdir():
        if not entry.is_dir():
            continue
        m = BATCH_RE.match(entry.name)
        if m:
            groups.setdefault(m.group("condition"), []).append(entry)
    return groups


def merge_condition(canonical: Path, batches: list[Path], dry_run: bool) -> None:
    canonical.mkdir(parents=True, exist_ok=True)
    ranking_path = canonical / "ranking_scores.csv"
    header = ["seed", "sample", "ranking_score"]
    rows = []
    if ranking_path.exists():
        with ranking_path.open() as f:
            rows = list(csv.reader(f))[1:]

    for batch in sorted(batches):
        seed_dirs = list(batch.glob("seed-*_sample-*"))
        print(f"  {batch.name}: moving {len(seed_dirs)} seed dirs")
        for sd in seed_dirs:
            dest = canonical / sd.name
            if dest.exists():
                print(f"    SKIP (already exists): {sd.name}")
                continue
            if not dry_run:
                shutil.move(str(sd), str(dest))

        batch_ranking = batch / "ranking_scores.csv"
        if batch_ranking.exists():
            with batch_ranking.open() as f:
                rows.extend(list(csv.reader(f))[1:])

        if not dry_run:
            shutil.rmtree(batch)

    if not dry_run:
        with ranking_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
    print(f"  -> {canonical}: {len(rows)} total ranking rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True, type=Path,
                     help="e.g. outputs/af3_wt")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    groups = find_batches(args.output_dir)
    if not groups:
        print("No batch folders found (nothing matching *_sSTART_END).")
        return

    for condition, batches in sorted(groups.items()):
        canonical = args.output_dir / condition
        print(f"{condition}: {len(batches)} batch folder(s) -> {canonical}")
        merge_condition(canonical, batches, args.dry_run)


if __name__ == "__main__":
    main()
