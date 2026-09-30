# Changelog

All notable changes to this repository are listed here, newest first.
The repository has no tagged releases; entries are dated by their commits.

## Fixes (2026-09-30)

No archived result, run log, model file or computed number changed. The
regenerated tables and macro files are byte-identical to those of the
previous code.

- `scripts/gen_tables.py`: before, wrote its 7 LaTeX files to `~/work/paper/`
  and stopped with `FileNotFoundError` if that folder did not exist. Now it
  writes the same files, with the same names and contents, to
  `results/tables/` and creates the folder.
- `scripts/fig_advantage.py`: before, wrote `adv_macros.tex` to
  `~/work/paper/` (same error if missing). Now it writes it to
  `results/tables/` and creates the folder. The figure is unchanged.
- `scripts/campaign_main.sh`, `campaign2.sh`, `resume_all.sh`: before, began
  with `cd /root/work/gcnsac` and, if that folder did not exist, printed an
  error and carried on in the current folder. Now they change to the
  repository folder (the folder above `scripts/`), or to `$GCNSAC_DIR` if
  that is set, and stop if the folder does not exist. They now also create
  `results/` if it is missing; before, no run started without it (each
  run's output goes to `results/<name>.log`).
- `gcnsac/surrogate.py`: before, the calibration cache was the fixed folder
  `~/work/gcnsac/results/cal/` in the home directory, outside the
  repository (created if missing), so README Way B had to copy the
  archived calibration files there. Now it is the repository's
  `results/cal/` (the folder above `gcnsac/`, where `run.py` also writes
  its results), or the folder named by the new environment variable
  `GCNSAC_CAL_DIR`. For a repository at `~/work/gcnsac`, as in the
  original setup, this is the same folder as before. Checked: README
  Way B without the copy step gives the same history as the archived run
  `tr_CT1_ptm65_sc_s0` (best FoM 3.792502335347418), leaves the archived
  `results/cal/ptm65.json` unchanged, and creates nothing under `~/work`;
  a missing calibration is redone and saved in `results/cal/` also when
  the script is started from another folder.
- New `.gitignore`: `results/`, `pdk/`, `__pycache__/`.
- New `scripts/check_pdk_reproduction.py`: re-simulates the stored best
  design of 10 archived runs (PTM 65 nm, PTM 45 nm, SKY130, GF180MCU) and
  reports match or mismatch; with `--gf180-pid` it also redoes the GF180MCU
  calibration with the archived ngspice process IDs.
- README: PDK download commands now pin GF180MCU commit
  `9f992d5a9186d1f7820c58f039c484ad35b2edea` and SKY130 commit
  `f62031a1be9aefe902d6d54cddd6f59b57627436`, with the evidence; the
  `mkdir -p ~/work/paper`, `mkdir -p ~/work/gcnsac/results/cal` and
  `mkdir -p results` steps and the `~/work/paper` paths are gone; the
  "Caveats" and "Notes" sections describe the new behaviour.
- README, GF180MCU: the earlier README said the GF180MCU files or ngspice
  settings used for the logs were unknown. Found instead: the GF180MCU model
  files have not changed since 2022, and `design.ngspice` switches on random
  transistor mismatch, which ngspice 42 seeds with its process ID. With the
  pinned files, ngspice 42 and the right process IDs, the archived
  `results/cal/gf180.json` is reproduced exactly; the process IDs of the
  GF180MCU run simulations are not recorded, so those logs cannot be
  recomputed exactly. The code was not changed. Details in README Section 10.

## Documentation update (30 September 2026)

Documentation only; no code, data, model or result file changed.

- README rewritten as a step-by-step guide: what the code does in plain
  language, the contents of `pdk_setup/` and of both archives, installation
  of ngspice and the open PDKs (with the `GCNSAC_PDK_ROOT` setting and the
  correct `pdk_setup/` paths), three ways to use the code, a table of all
  scripts with measured run times, which script makes which figure and table,
  the values fixed in the code, and known caveats (fixed `~/work/...`
  folders, `cd /root/work/gcnsac` in the shell scripts, GF180MCU results not
  matching the current upstream model files).
- Corrected statements of the earlier README: the model archive holds 239
  `.pt` weight files and 209 `.npy` adaptation recordings (not "448 trained
  model checkpoints"); `campaign_main.sh` runs 10 methods with seeds 0-2 (the
  remaining method and seeds come from `chunk2.py` and `chunk3.py`).
- Added this CHANGELOG and a CITATION.cff.

## Initial upload (24 July 2026)

From the git history (commits `616731d` to `58d04fb`):

- README with the project title, Apache-2.0 LICENSE, and the run-log archive
  `results_logs.tar.gz`.
- Trained weights as `models.tar.gz.part-00` to `part-09`.
- The `gcnsac/` package and the `scripts/` folder.
- `pdk_setup/` with the PTM 65 nm and 45 nm model cards and the SKY130
  wrapper `sky130_mini_tt.spice`.
