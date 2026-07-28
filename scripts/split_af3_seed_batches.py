#!/usr/bin/env python3
"""Split AF3 modelSeeds JSON inputs into smaller seed-range batches.

Each output JSON keeps the same sequences/dialect/version but carries only a
slice of modelSeeds, and its "name" field is suffixed with the seed range
(e.g. 1A52_WT_apo_s251_500) so AF3 writes each batch to its own output
subfolder instead of racing with sibling array tasks on shared summary files.
"""
import argparse
import json
from pathlib import Path


def split_file(src: Path, out_dir: Path, batch_size: int) -> list[str]:
    data = json.loads(src.read_text())
    seeds = data["modelSeeds"]
    written = []
    for i in range(0, len(seeds), batch_size):
        chunk = seeds[i : i + batch_size]
        batch = dict(data)
        batch["modelSeeds"] = chunk
        batch["name"] = f"{data['name']}_s{chunk[0]}_{chunk[-1]}"
        out_path = out_dir / f"{batch['name']}.json"
        out_path.write_text(json.dumps(batch, indent=2))
        written.append(out_path.name)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=250)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for src in sorted(args.input_dir.glob("*.json")):
        names = split_file(src, args.output_dir, args.batch_size)
        print(f"{src.name}: {len(names)} batches")
        for n in names:
            print(f"  {n}")
        total += len(names)
    print(f"\nWrote {total} batch JSON files to {args.output_dir}")


if __name__ == "__main__":
    main()
