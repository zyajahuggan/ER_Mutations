# ER_Mutations

Collaboration project with the Nayar Lab studying how two clinically relevant
mutations in the estrogen receptor alpha (ERα) ligand-binding domain —
**H524L** and **Y537S** — affect receptor conformation, dimerization, ligand
binding, and coactivator recruitment. Y537S in particular is a well-known
constitutively-activating ESR1 mutation recurrent in metastatic,
endocrine-therapy-resistant breast cancer; H524L is studied here for
comparison. The project combines Rosetta deep mutational scanning (DMS),
AlphaFold3 (AF3) structure prediction, and downstream conformational/interface
analysis to characterize how each mutation shifts the receptor's
active/inactive conformational equilibrium.

## Repository layout

```
inputs/     structure templates, ligand params, and AF3 job inputs
scripts/    all pipeline code, grouped by stage
outputs/    everything the pipeline produces, mirroring scripts/
archive/    retired repo-root artifacts (old job-launcher scripts)
logs/       SLURM stdout/stderr
repos/      vendored external code (not authored by this project)
```

### `inputs/`

| Folder | Contents |
|---|---|
| `1A52/`, `1ERE/`, `1GWR/` | Three ERα LBD crystal structures used as DMS scaffolds. Each holds the raw downloaded file plus 4 cleaned variants (`_apo`, `_dimer`, `_dimer_apo`, holo) produced by `scripts/clean/`. `1A52` was the original/default template; `1ERE` and `1GWR` were added later as alternate scaffolds — note only `1ERE`/`1GWR` have Y537S runs (see `outputs/dms/`). |
| `est_ligand/` | Rosetta params (`EST.params`) and source ligand files (`EST.pdb`, `EST.sdf`) for 17β-estradiol, generated once via `scripts/clean/gen_EST_params.sh` and reused by every DMS/interface/coactivator run that scores the bound ligand. |
| `active_inactive_refs/` | `3erd.cif` (agonist + coactivator peptide bound → the "active" H12 conformation) and `3ert.cif` (antagonist bound → "inactive"), the reference structures the active/inactive classification analyses are built against. |
| `af3_json/` | Batched AF3 input JSONs, one set per genotype (WT/H524L/Y537S) × seed range × monomer/dimer, consumed by the (now-archived) AF3 SLURM launcher scripts. |

### `scripts/`

| Folder | Contents |
|---|---|
| `clean/` | `clean_1A52.py` / `clean_1ERE.py` / `clean_1gwr.py` — parse a raw PDB/CIF template and emit the cleaned apo/holo/monomer/dimer PDBs everything downstream depends on. `gen_EST_params.sh` — one-time Rosetta ligand-params generation for estradiol. |
| `dms/` | Deep mutational scanning at H524 and Y537. 16 `dms_*.py` variants (H524/Y537 × apo/holo × monomer/dimer × base/1GWR) each run Rosetta FastRelax + `InterfaceAnalyzerMover`, paired with a matching `.xml` relax protocol. 18 `submit_dms_*.sh` SLURM wrappers drive them end-to-end: clean the structure if it doesn't exist yet → check `EST.params` exists → launch the DMS. |
| `interface/` | `interface_analyzer*.py` score the DMS output PDBs' receptor–receptor dimer interface and receptor–ligand interface (`dG_interface`, `dSASA`, `sc`, `packstat`, `unsat_hbonds`); `plot_interface_metrics.py` visualizes the results. |
| `coactivator/` | `coactivator_analyzer.py` grafts the GRIP1/SRC-2 coactivator peptide (`LXXLL` motif, sourced from 3ERD) onto the best WT/H524L/Y537S apo+holo DMS structures and scores that separate AF-2-groove interface — a different surface from the dimer/ligand interfaces above. `rescore_coactivator_reps.py` / `plot_coactivator_scores.py` support and visualize it. |
| `af3/` | The AlphaFold3-based conformational analysis pipeline: `af3_conformations.py` (per-condition and pooled clustering, plus active/inactive classification against the 3ERD/3ERT references), `af3_pca.py` (global PCA across all conditions), `af3_cluster_sample.py`, `cluster_confidence.py` (AF3 confidence metrics averaged per cluster), `consolidate_and_rank_H524L.py`, and AF3 job-management utilities (`merge_af3_batches.py`, `split_af3_seed_batches.py`, `run_af3local_array_job.sh`). |

### `outputs/`

| Folder | Contents |
|---|---|
| `af3/` | Raw AlphaFold3 prediction outputs, one folder per genotype: `af3_wt`, `af3_H524L`, `af3_Y537S`, `af3_wt_monomer` — each holding one subfolder per seed/sample. |
| `dms/` | Rosetta DMS outputs, one folder per structural template: `dms_1A52/` (H524 only — no Y537S was ever run against the raw 1A52 template), `dms_1ERE/`, `dms_1GWR/` (both H524 and Y537 variants, plus their interface-score CSVs/heatmaps). |
| `analysis/` | Downstream conformational/interface analysis built on top of `af3/`. Currently holds one live analysis, `cluster_vs_active_whole_2026-07-22/` (see below); every earlier/superseded iteration lives in `analysis/archive/`, renamed with a description of what it did and the date it was generated. |
| `archive/` | Outputs-level items retired outside of `analysis/` — currently the coactivator peptide-graft structures (`coactivator_graft_3erd_onto_1ere_2026-07-21/`). |

**The current analysis** — `outputs/analysis/cluster_vs_active_whole_2026-07-22/`
pools every dimer variant/state's AF3 predictions together, aligns each onto
the active (3ERD) reference structure, clusters them (k-means/DBSCAN, k picked
by silhouette) over the *whole* aligned structure (not just H12), and reports
which cluster sits closest to "active" plus what fraction of each
variant/state falls into each cluster. Earlier iterations preserved in
`analysis/archive/` (PCA-only, per-condition-only, H12-region-only,
hand-picked-residue-window versions) were superseded by this approach as the
analysis matured — see each folder's date/name for what it did differently.

### `archive/` (repo root)

`af_cpu_jobs/` — 68 old AlphaFold3 SLURM/singularity job-launcher scripts,
archived once their jobs had completed and merged into `outputs/af3/`.

### `repos/`

Vendored external research code, not authored by this project —
currently [`bioemu`](https://github.com/microsoft/bioemu), Microsoft's protein
conformational ensemble generator.

## Typical pipeline order

1. `scripts/clean/gen_EST_params.sh` (once) — generate `EST.params` for estradiol.
2. `scripts/clean/clean_1A52.py` / `clean_1ERE.py` / `clean_1gwr.py` — produce cleaned structures (or let a `submit_dms_*.sh` script generate them automatically if missing).
3. `scripts/dms/submit_dms_*.sh` — SLURM DMS at H524/Y537 for a given template + apo/holo + monomer/dimer.
4. `scripts/interface/interface_analyzer*.py` — score dimer/ligand interfaces from the DMS output.
5. `scripts/coactivator/coactivator_analyzer.py` — test coactivator (AF-2 groove) binding on the top-ranked DMS structures.
6. AlphaFold3 predictions (run externally against the `inputs/af3_json/` batches) → `scripts/af3/merge_af3_batches.py` to consolidate seed batches.
7. `scripts/af3/af3_conformations.py`, `af3_pca.py`, `cluster_confidence.py` — conformational clustering, active/inactive classification, and AF3-confidence QC on the merged predictions.
