# Changelog

All notable changes to this repository are listed here, newest first.
The repository has no tagged releases; entries are dated by their commits.

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
