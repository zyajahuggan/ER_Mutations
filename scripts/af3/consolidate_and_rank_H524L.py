"""
Adapted from consolidate_and_rank.py for ER_Mutations / H524L AF3 outputs.

Layout:
  af3_H524L/
    {variant}/          e.g. 1a52_h524l_apo, 1a52_h524l_holo
      seed-{s}_sample-{p}/
        summary_confidences.json
        confidences.json
        model.cif

Outputs (under af3_H524L/analysis/):
  metrics_all.csv           — every prediction row
  metrics_best_per_seed.csv — best AF3 sample per seed (per variant)
  top3_summary.csv          — top-3 per variant
  top3_per_variant/         — CIF + JSONs for top-3 per variant
"""

import json
import re
import shutil
import csv
from pathlib import Path

AF3_OUT  = Path("/home/zhuggan1/scr4_jgray21/zhuggan1/repos/ER_Mutations/outputs/af3_Y537S")
ANALYSIS = AF3_OUT / "analysis"
TOP3_DIR = ANALYSIS / "top3_per_variant"

SCALAR_KEYS = ["ranking_score", "iptm", "ptm", "fraction_disordered", "has_clash"]


def load_summary(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def parse_all_predictions() -> list[dict]:
    rows = []
    for variant_dir in sorted(AF3_OUT.iterdir()):
        if not variant_dir.is_dir() or variant_dir.name == "analysis":
            continue
        variant = variant_dir.name
        for pred_dir in sorted(variant_dir.iterdir()):
            m = re.match(r"seed-(\d+)_sample-(\d+)$", pred_dir.name)
            if not m or not pred_dir.is_dir():
                continue
            seed, sample = int(m.group(1)), int(m.group(2))
            sc_file  = pred_dir / "summary_confidences.json"
            cif_file = pred_dir / "model.cif"
            if not sc_file.exists():
                continue
            sc = load_summary(sc_file)
            row = {
                "variant":    variant,
                "seed":       seed,
                "af3_sample": sample,
                "pred_dir":   str(pred_dir),
                "cif_path":   str(cif_file) if cif_file.exists() else "",
            }
            for k in SCALAR_KEYS:
                row[k] = sc.get(k, None)
            row["chain_ptm"]  = ";".join(str(v) for v in sc.get("chain_ptm",  []))
            row["chain_iptm"] = ";".join(str(v) for v in sc.get("chain_iptm", []))
            rows.append(row)
    return rows


def rank_metric(variant: str) -> str:
    """iptm for holo structures, ptm for apo structures."""
    return "iptm" if "holo" in variant.lower() else "ptm"


def best_per_seed(rows: list[dict]) -> list[dict]:
    """For each (variant, seed) keep the sample with highest iptm (holo) or ptm (apo)."""
    groups: dict[tuple, list] = {}
    for row in rows:
        key = (row["variant"], row["seed"])
        groups.setdefault(key, []).append(row)
    best = []
    for (variant, seed), group in groups.items():
        metric = rank_metric(variant)
        valid = [r for r in group if r[metric] is not None]
        if not valid:
            continue
        best.append(max(valid, key=lambda r, m=metric: r[m]))
    return best


def top3_per_variant(best_rows: list[dict]) -> dict[str, list[dict]]:
    """Return {variant: [top3 rows sorted desc by iptm (holo) or ptm (apo)]}."""
    groups: dict[str, list] = {}
    for row in best_rows:
        groups.setdefault(row["variant"], []).append(row)
    return {
        v: sorted(rows, key=lambda r, m=rank_metric(v): r[m], reverse=True)[:3]
        for v, rows in groups.items()
    }


def write_csv(rows: list[dict], path: Path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows → {path}")


def copy_top3(top3: dict[str, list[dict]]):
    TOP3_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for variant, rows in sorted(top3.items()):
        for rank, row in enumerate(rows, start=1):
            tag  = f"{variant}_rank{rank}"
            dest = TOP3_DIR / tag
            dest.mkdir(exist_ok=True)
            src  = Path(row["pred_dir"])
            for fname in ["model.cif", "summary_confidences.json", "confidences.json"]:
                src_file = src / fname
                if src_file.exists():
                    shutil.copy2(src_file, dest / fname)
            summary_rows.append({
                "rank_in_variant": rank,
                **{k: row[k] for k in ["variant", "seed", "af3_sample",
                                        "ranking_score", "iptm", "ptm",
                                        "fraction_disordered", "has_clash",
                                        "chain_ptm", "chain_iptm"]},
                "dest": str(dest),
            })
            metric = rank_metric(variant)
            print(f"  {variant} rank={rank}  {metric}={row[metric]:.4f}  → {tag}")
    write_csv(summary_rows, ANALYSIS / "top3_summary.csv")


def main():
    print("Parsing all AF3 predictions …")
    all_rows = parse_all_predictions()
    print(f"  found {len(all_rows)} predictions total")

    write_csv(all_rows, ANALYSIS / "metrics_all.csv")

    best = best_per_seed(all_rows)
    best_sorted = sorted(best, key=lambda r: (r["variant"], r["seed"]))
    write_csv(best_sorted, ANALYSIS / "metrics_best_per_seed.csv")

    top3 = top3_per_variant(best)
    print("\nTop-3 per variant:")
    copy_top3(top3)

    print("\nDone. Outputs written to:", ANALYSIS)


if __name__ == "__main__":
    main()
