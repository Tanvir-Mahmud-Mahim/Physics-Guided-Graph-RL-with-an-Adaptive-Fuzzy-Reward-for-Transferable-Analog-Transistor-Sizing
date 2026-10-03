# Physics-Guided Graph RL: details

This file has the full notes for this repository. For the overview, installation and quick start,
see [README.md](README.md).

## Contents

- [Extra notes for Sections 1 to 4](#extra-notes-for-sections-1-to-4)
  - [More on Section 2: The files and archives](#more-on-section-2-the-files-and-archives)
  - [More on Section 3: Installation](#more-on-section-3-installation)
- [5. The scripts, step by step](#5-the-scripts-step-by-step)
- [6. Which script makes which figure and table](#6-which-script-makes-which-figure-and-table)
- [7. The Python modules](#7-the-python-modules)
- [8. Where the numbers come from](#8-where-the-numbers-come-from)
- [9. Built-in checks](#9-built-in-checks)
- [10. Notes on the calculations](#10-notes-on-the-calculations)
- [11. Version history](#11-version-history)

---

## Extra notes for Sections 1 to 4

### More on Section 2: The files and archives

```
Physics-Guided-Graph-RL-...-Transistor-Sizing/
|-- README.md              this guide
|-- CHANGELOG.md           what changed, newest first
|-- CITATION.cff           citation details (drives the "Cite this repository" button)
|-- LICENSE                Apache-2.0 license
|-- .gitignore             keeps results/, pdk/ and __pycache__/ out of git
|-- gcnsac/                the Python package (see Section 7)
|   |-- pdk.py             the four technologies: model file paths, supply voltage, size limits
|   |-- circuits.py        the three test circuits: netlists and circuit graphs
|   |-- env.py             runs ngspice and computes the unified figure of merit
|   |-- tskf.py            the adaptive interval type-2 fuzzy reward
|   |-- surrogate.py       the simple physics model and its calibration against ngspice
|   |-- agents.py          GCN-SAC (proposed), GCN-DDPG, A2C, PPO
|   `-- baselines.py       Bayesian optimization (BO) and MACE-style BO
|-- scripts/
|   |-- run.py             one optimization run (one method, circuit, technology, seed)
|   |-- campaign_main.sh   main comparison, seeds 0-2
|   |-- campaign2.sh       ablations and transfer, seeds 0-2 (waits for campaign_main.sh)
|   |-- resume_all.sh      the two campaigns above, skipping runs that already exist
|   |-- chunk.py           the same runs as the two campaigns, in time-limited batches
|   |-- chunk2.py          extension: seeds 3-4, EKV on CT-2, 1500-evaluation pretraining, transfer from it
|   |-- worker.py          looping worker for the chunk2.py job list
|   |-- chunk3.py          EKV version on CT-1 and CT-3, and transfer from EKV-trained agents
|   |-- aggregate.py       collects all run logs into results/agg/aggregate.json
|   |-- gen_tables.py      LaTeX tables and text macros for the paper (into results/tables/)
|   |-- figures.py         paper figures (PDF)
|   |-- fig_advantage.py   the "advantage" figure and its text macros
|   `-- check_pdk_reproduction.py  checks that your ngspice and PDK files reproduce the archived logs
|-- pdk_setup/             helper model files (no foundry PDK files, see below)
|   |-- 65nm_bulk.pm       PTM 65 nm model card (text)
|   |-- 45nm_bulk.pm       PTM 45 nm metal-gate/high-k model card (text)
|   `-- sky130_mini_tt.spice  23-line wrapper that loads the SKY130 typical models
|-- results_logs.tar.gz    run logs, aggregate file, calibration, normalization, figures (0.7 MB)
`-- models.tar.gz.part-00 ... part-09   trained network weights, split into 10 parts (86 MB packed, 95 MB unpacked)
```

**What `pdk_setup/` contains.** You will find two PTM model cards (`65nm_bulk.pm`, 146
lines, with the comments "PTM 65nm NMOS" and "PTM 65nm PMOS"; `45nm_bulk.pm`,
141 lines, headed "PTM High Performance 45nm Metal Gate / High-K /
Strained-Si") and a small wrapper,
`sky130_mini_tt.spice`. The wrapper holds no model data. It includes four
files from the official SKY130 repository (`parameters/invariant.spice`,
`parameters/lod.spice`, and the typical-corner `nfet_01v8` and `pfet_01v8`
model files). It also sets 18 parameters to zero (9 per transistor type, with
names ending in `_slope`, `_slope1` or `_slope_spectre`).

**What the archives contain.** I listed their contents with `tar tzvf`, and nothing else is inside:

| Archive | Unpacks to | Contents |
|---|---|---|
| `results_logs.tar.gz` | `results/*.json` | 326 run logs (one per run) |
| | `results/agg/aggregate.json` | summary of all runs (made by `aggregate.py`) |
| | `results/cal/*.json` | 4 calibration files of the physics model (one per technology) |
| | `results/norm/*.json` | 12 normalization files (3 circuits x 4 technologies) |
| | `results/figs/*.pdf` | 8 figures |
| | `results/policy_ft/*.json` | 27 further fine-tuning logs named like the `tr_*_ft` runs; no script in this repository reads or writes this folder |
| `models.tar.gz.part-*` | `results/w/*.pt` | 239 PyTorch weight files of the graph-based agents |
| | `results/w/*_tskfhist.npy` | 209 recordings of the fuzzy reward's adaptation |

The earlier README described the model archive as "448 trained model
checkpoints (`results/w/*.pt`)". In fact, the 448 files are 239 `.pt` files
plus 209 `.npy` files. Neither archive contains PDK or model-card files.

### More on Section 3: Installation

These are the versions I used for the checks in this guide: Python 3.11.15, PyTorch 2.14.0
(only the CPU is used), NumPy 2.4.4, SciPy 1.17.1, Matplotlib 3.10.9.

**Which PDK version?** The repository did not record the versions used for
the run logs. I found and checked these pins as follows (September 2026):

| PDK | Commit | Evidence |
|---|---|---|
| GF180MCU | `9f992d5a9186d1f7820c58f039c484ad35b2edea` (31 May 2023, current `main`) | The two model files have never changed on `main`: `design.ngspice` (SHA-256 `8d9721a5...`) and `sm141064.ngspice` (SHA-256 `73fc67d3...`) are identical in all four commits that touch `models/ngspice` (`92ec4b2`, 20 July 2022, to `13fe03d`), so any commit from `92ec4b2` on gives the same files. With them, ngspice 42 reproduces the archived calibration file `results/cal/gf180.json` bit for bit, once ngspice gets the process IDs described in Section 10. |
| SKY130 | `f62031a1be9aefe902d6d54cddd6f59b57627436` (current `main`) | Re-simulated best designs and the calibration match the archived logs exactly. |
| PTM 65 nm / 45 nm | the cards in `pdk_setup/` | Re-simulated best designs match exactly. |

---

## 5. The scripts, step by step

| Step | Command | What it does | Time* | Results |
|---|---|---|---|---|
| 0 | `tar xzf results_logs.tar.gz` and `cat models.tar.gz.part-* \| tar xz` | Unpacks the archived runs and weights | 1.5 s | `results/` |
| 1 | `python3 scripts/run.py --method M --ct CT --pdk P --seed S --budget B` | One optimization run (see the options below) | 27 s for 50 BO simulations and 65-115 s for 150 simulations of the proposed method (CT-1, 65 nm); 214 s for 600 A2C simulations of CT-3 at GF180MCU. The logs record 46-593 s for 600-simulation runs | `results/<tag>.json`; graph agents also `results/w/<tag>.pt` and, with the fuzzy reward, `results/w/<tag>_tskfhist.npy` |
| 2 | `bash scripts/campaign_main.sh` | Main comparison on GF180MCU: 10 methods x 3 circuits x seeds 0-2, 600 simulations each (90 runs) | long (not re-timed); logs: 4.8 h of run time | `results/*.json`, `results/campaign_main.log` |
| 3 | `bash scripts/campaign2.sh` | Waits for step 2, makes the normalization files for SKY130/PTM65/PTM45, then: 4 ablations on CT-2 (seeds 0-2); technology transfer to 3 nodes for 3 circuits, fine-tuned (`tr_*_ft`) and from scratch (`tr_*_sc`); topology transfer CT-2 to CT-3 and back (`tt_*`); 150 simulations each (78 runs) | long (not re-timed); logs: 2.4 h | as above, `results/campaign2.log` |
| 2+3 | `bash scripts/resume_all.sh` | Same runs as steps 2 and 3, skipping any run whose log already exists | as steps 2+3 | `results/campaign_resume.log` |
| 2+3 | `python3 scripts/chunk.py [seconds]` | Same 168 runs as steps 2 and 3, but runs only one batch that fits in the given time (default 480 s) on two lanes; call it again until it prints `ALL_RUNS_COMPLETE` | as steps 2+3 | `results/*.json` |
| 4 | `python3 scripts/chunk2.py` | Extension (115 runs): seeds 3-4 for all 10 methods and 4 ablations; the EKV version (`gcnsac_tskf_pia_ekv`) on CT-2 with seeds 0-4; 1500-simulation pretraining (`pia1500_*`, seeds 0-2); transfer from it with only the graph encoder reused (`tr2_*`, `tt2_*`). Modes: no argument = run all on two lanes; `count` = count missing runs; a number = one time-limited batch | long (not re-timed); logs: 7.8 h | `results/*.json` |
| 4 | `python3 scripts/worker.py [name]` | Alternative to step 4: a worker that takes the next free `chunk2.py` job (several workers can run at once) | as step 4 | `results/claims/` (lock files) |
| 5 | `python3 scripts/chunk3.py worker [name]` | EKV version on CT-1 and CT-3 (seeds 0-4), and transfer from the EKV-trained agents (`tr3_*`, `tt3_*`). `count` mode counts ready and blocked runs | long (not re-timed); logs: 1.5 h | `results/*.json`, `results/claims3/` |
| 6 | `python3 scripts/aggregate.py` | Collects all logs into one summary and prints the main comparison | 0.8 s | `results/agg/aggregate.json` |
| 7 | `python3 scripts/gen_tables.py` | Writes the paper's LaTeX tables and text macros | 2.6 s | `results/tables/*.tex` |
| 8 | `python3 scripts/figures.py` | Draws the paper figures | 8.4 s | `results/figs/*.pdf` |
| 9 | `python3 scripts/fig_advantage.py` | Draws the advantage figure and writes its macros | 1.9 s | `results/figs/F_advantage.pdf`, `results/tables/adv_macros.tex` |
| check | `python3 scripts/check_pdk_reproduction.py [--pdk ...] [--gf180-pid]` | Re-simulates the stored best design of 10 archived runs (2-3 per technology) and compares metrics and FoM with the logs; `--gf180-pid` also redoes the GF180MCU calibration with the archived ngspice process IDs (Section 9) | 22 s for the 10 designs; 30 s for `--pdk gf180 --gf180-pid` | printed report; exit status 0 if all expected matches are found |

\*I measured these times on a shared two-core computer. The exception is "logs", which is the
sum of the `wall_s` field over the archived runs of that step (single-run
times on my machine).

**Options of `run.py`.** `--method` is one of `bo`, `mace`, `a2c`, `ppo`,
`a2c_tskf`, `ppo_tskf`, `gcnddpg`, `gcnsac`, `gcnsac_tskf`,
`gcnsac_tskf_pia`, `gcnsac_tskf_pia_ekv`, or an ablation
`gcnsac_tskf_pia_noper` / `_noaug` / `_nosimclr` / `_t1`. The parts of the
name switch features on or off: `tskf` = fuzzy reward, `pia` = physics
guidance (square-law model; `ekv` = EKV model), `noper` = no prioritized
replay, `noaug` = no graph augmentation, `nosimclr` = no contrastive loss,
`t1` = type-1 fuzzy reward. `--ct` is `CT1`, `CT2` or `CT3`; `--pdk` is
`gf180` (default), `sky130`, `ptm65` or `ptm45`; `--budget` is the number of
simulations (default 600). `--load FILE.pt` starts from saved weights
(fine-tuning; the encoder then learns 10 times slower, `--enc_lr_scale`),
`--enc_only` reuses only the graph encoder, and `--tag` names the output
files.

**Order matters.** Transfer runs load weights made by earlier runs:
`tr_`/`tt_` need `results/w/gcnsac_tskf_pia_<CT>_gf180_s<seed>.pt` (step 2),
`tr2_`/`tt2_` need `results/w/pia1500_<CT>_s<seed>.pt` (step 4), and
`tr3_`/`tt3_` need `results/w/gcnsac_tskf_pia_ekv_<CT>_gf180_s<seed>.pt`
(steps 4 and 5). `chunk2.py` and `chunk3.py` skip a transfer run until its
weights exist.

**Caveats found in the scripts.**

- `campaign_main.sh`, `campaign2.sh` and `resume_all.sh` first change to
  the repository folder (the folder above `scripts/`), wherever they are
  started from. To use another folder, set `GCNSAC_DIR=/path/to/folder`. If
  that folder does not exist, the script stops.
- These three scripts send each run's output to `results/<name>.log` and
  create `results/` if it does not exist yet.
- `chunk.py`, `chunk2.py`, `chunk3.py` and `worker.py` use relative paths,
  so you must also start them from the repository folder.
- If a normalization file is missing, `run.py` first simulates 300 random
  designs to make it (`norm_sims` in the log). All archived logs have
  `norm_sims` = 0, because the files already existed.

---

## 6. Which script makes which figure and table

The repository does not record the paper's figure numbers. The one exception is that
the earlier README called `F_advantage.pdf` "Fig. 3". So the table lists the
output files.

| Output file | Content (as drawn by the code) | Data used | Drawn by |
|---|---|---|---|
| `results/figs/F5_learning.pdf` | Best FoM so far versus simulations for BO, MACE, GCN-DDPG, GCN-SAC, GCN-SAC-TSKF and the EKV version, on the three circuits | 600-simulation GF180MCU logs | `figures.py` (`fig_learning`) |
| `results/figs/F6_tskf.pdf` | Fuzzy-reward adaptation (error, size of the rule outputs, mean rule firing) for each circuit | `results/w/gcnsac_tskf_pia_<CT>_gf180_s0_tskfhist.npy` | `figures.py` (`fig_tskf`) |
| `results/figs/F6_convergence.pdf` | First 200 simulations of GCN-SAC, GCN-SAC-TSKF and the EKV version on CT-1 and CT-2 | 600-simulation GF180MCU logs | `figures.py` (`fig_convergence`) |
| `results/figs/F8a_tech_transfer.pdf` | Technology transfer (with and without transfer) for 3 circuits x 3 target nodes, plus two topology-transfer panels | `tr3_*_ft` (or `tr_*_ft` if fewer than three), `tr_*_sc`, `tt_*` | `figures.py` (`fig_transfer`) |
| `results/figs/F8b_topo_transfer.pdf` | Topology transfer | `tt3_*_ft` (or `tt_*_ft`), `tt_*_sc` | `figures.py` (`fig_topo`), **but this function is not called** when `figures.py` runs, so only the archived copy of this file exists |
| `results/figs/F9_ablation.pdf` | Ablation study on CT-2 | `results/agg/aggregate.json` | `figures.py` (`fig_ablation`) |
| `results/figs/FS1_tskf_dynamics.pdf` | Supplementary: fuzzy-reward adaptation traces | same `.npy` files as `F6_tskf.pdf` | `figures.py` (`fig_tskf_supp`) |
| `results/figs/F_advantage.pdf` | (a) final FoM of the EKV version, GCN-DDPG and BO; (b) simulations the EKV version needs to reach GCN-DDPG's final FoM | 600-simulation GF180MCU logs | `fig_advantage.py` |
| `results/tables/tab_fom.tex` | Main FoM table (mean and spread, 11 methods, 3 circuits) | `aggregate.json` | `gen_tables.py` |
| `results/tables/tab_metrics.tex` | Measured gain, bandwidth, power and noise of each method's best design | `aggregate.json` | `gen_tables.py` |
| `results/tables/tab_transfer.tex`, `tab_transfer_full.tex` | Transfer results (fine-tuned versus from scratch; the full version adds topology transfer) | `aggregate.json` | `gen_tables.py` |
| `results/tables/tab_perseed.tex` | Per-seed FoM and Welch t-test against the three strongest other methods | run logs | `gen_tables.py` |
| `results/tables/tab_cost.tex` | Simulations and mean run time per method | `aggregate.json` | `gen_tables.py` |
| `results/tables/results_macros.tex`, `adv_macros.tex` | Numbers and sentences used in the paper text | `aggregate.json`, run logs | `gen_tables.py`, `fig_advantage.py` |

I compared the regenerated figures with the archived PDFs by rendering
both to images. All seven that `figures.py` and `fig_advantage.py` redraw
are pixel-identical.

---

## 7. The Python modules

| File | What it contains |
|---|---|
| `gcnsac/pdk.py` | The four technologies: model file paths (under `GCNSAC_PDK_ROOT`), supply voltage, transistor names, smallest and largest W and L, and the sizing grid |
| `gcnsac/circuits.py` | The three circuits: their design variables, the ngspice netlist with the measurements, and the circuit graph (device types, connections, stage numbers) |
| `gcnsac/env.py` | Writes a netlist, runs `ngspice -b` (10 s time limit), reads the results, and computes the unified FoM; also makes the normalization files from random designs |
| `gcnsac/tskf.py` | The interval type-2 fuzzy reward: 3 fuzzy sets per metric, 3^n rules, Karnik-Mendel type reduction (the standard way to get one number out of an interval type-2 system), and online adaptation of the rule outputs |
| `gcnsac/surrogate.py` | Calibration of a simple transistor model (threshold voltage, current factor, output-resistance factor, EKV slope factor) from ngspice sweeps, and the differentiable physics FoM used for guidance |
| `gcnsac/agents.py` | Graph builder and graph augmentation, GCN encoder, SimCLR contrastive loss, prioritized replay (PER), GCN-SAC, GCN-DDPG, and A2C/PPO on a plain vector |
| `gcnsac/baselines.py` | A small Gaussian-process BO with expected improvement, and the MACE-style variant that picks three points per step (expected improvement, upper confidence bound, probability of improvement) |

---

## 8. Where the numbers come from

**Device models.** GF180MCU: 3.3 V devices `nmos_3p3` / `pmos_3p3`, section
`typical` of `sm141064.ngspice`. SKY130: 1.8 V devices
`sky130_fd_pr__nfet_01v8` / `pfet_01v8`, typical corner (through the wrapper
in `pdk_setup/`). PTM: the 65 nm and 45 nm cards in `pdk_setup/`, device names
`nmos` / `pmos`. The comments in `gcnsac/pdk.py` describe the nodes as
"GF180MCU, 180 nm", "SKY130, 130 nm", "ASU Predictive Technology Model, 65 nm
bulk CMOS" and "45 nm metal-gate/high-k".

**Size limits and supply** (from `gcnsac/pdk.py`):

| Node | Supply | L range | W range | Grid |
|---|---|---|---|---|
| `gf180` | 3.3 V | 0.28 to 4 um | 0.22 to 200 um | 10 nm |
| `sky130` | 1.8 V | 0.15 to 4 um | 0.42 to 100 um | 5 nm |
| `ptm65` | 1.1 V | 65 nm to 1 um | 0.13 to 50 um | 5 nm |
| `ptm45` | 1.0 V | 45 nm to 1 um | 0.09 to 50 um | 5 nm |

**Design variables** (from `gcnsac/circuits.py`). CT-1 has 10 (five widths,
three lengths, the Miller capacitor 0.1 to 10 pF, the bias current 1 to
50 uA). CT-2 has 10 and CT-3 has 13 (input device W and L, then W of the
NMOS and PMOS and a shared L per stage, the capacitor 0.05 to 5 pF, the bias
current 1 to 100 uA). Widths range from twice the smallest width of the node
to the largest, except the TIA input device (smallest width to 0.3 x the
largest). Lengths use the node's full L range. All values are set on a
logarithmic scale between their limits, and W and L are rounded to the grid.

**Fixed circuit values.** CT-1: 1 pF load, input common-mode voltage 0.42 x
supply (0.45 x supply for PTM), gain read at 10 Hz, noise integrated from
10 Hz to 10 MHz. CT-2 and CT-3: input current source with DC value 0.2 x the
bias current, 50 fF input capacitor, 100 fF load, gain (transimpedance) read
at 10 kHz, noise integrated from 1 kHz to 1 GHz.

**Measured metrics.** CT-1: gain (dB), unity-gain frequency, phase margin,
power, input-referred noise. CT-2 and CT-3: transimpedance, -3 dB bandwidth,
power, input-referred noise.

**Budgets and seeds** (from the scripts): 600 simulations per run in the main
comparison and the ablations (seeds 0-4); 150 simulations per transfer run
(seeds 0-2); 1500 for the `pia1500_*` pretraining (seeds 0-2); 300 random
designs (seed 1234) per circuit and node for normalization, shared by all
runs. The physics-model calibration uses 8 ngspice runs per node (2
gate-voltage sweeps and 6 drain-voltage sweeps). It is stored and reused, and
it is counted in `sims_used` only in the run that made it (these runs show
158 instead of 150). The earlier README and `gen_tables.py` say "24
calibration DC sweeps", but the code performs 8.

**Learning settings fixed in the code** (`agents.py`, `tskf.py`,
`baselines.py`, `run.py`):

| Part | Setting |
|---|---|
| Episodes | 4 simulations, then a restart near the middle of the design range |
| GCN encoder | 2 graph layers, 64 wide, 11 input features per device |
| SAC | learning rate 3e-4, batch 64, discount 0.5, target mixing 0.005, starting entropy weight 0.1 |
| Replay (PER) | up to 20000 samples, priority exponent 0.6, correction 0.4 |
| Augmentation / contrastive | 10% feature masking and 10% link dropping; SimCLR temperature 0.2, weight 0.1 |
| Physics guidance | weight 0.3, on up to 8 samples per update |
| Fuzzy reward | uncertainty width 0.15 (0 for the type-1 ablation), step 0.05, momentum 0.9 |
| Fine-tuning | encoder learning rate x 0.1 |
| GCN-DDPG | exploration noise 0.4, shrinking to 0.1 |
| A2C / PPO | state = previous design vector; PPO: 4 passes, clip 0.8 to 1.2 |
| BO / MACE | 40 random starting designs, 600 random candidates per step, refit every 5 simulations; MACE upper confidence bound factor 1.8 |

---

## 9. Built-in checks

The repository has no automated test suite. I ran these checks for this
guide:

- **Tables and figures from the archived logs** (Way A): `aggregate.py`
  rebuilds `results/agg/aggregate.json` byte for byte; the seven figures that
  `figures.py` and `fig_advantage.py` redraw are pixel-identical to the
  archived PDFs.
- **One run from scratch** (Way B): I re-ran `tr_CT1_ptm65_sc_s0`, and it gave
  the identical history, best FoM and best design.
- **Re-simulating stored designs** (`scripts/check_pdk_reproduction.py`,
  with ngspice 42 and the pinned PDK commits of Section 3.2): the best designs
  of `tr_CT1_ptm65_sc_s0`, `tr_CT3_ptm65_ft_s0` (PTM 65 nm),
  `tr_CT2_ptm45_sc_s0`, `tr_CT3_ptm45_ft_s1` (PTM 45 nm),
  `tr_CT1_sky130_ft_s2`, `tr_CT2_sky130_sc_s1` and `tr_CT3_sky130_sc_s0`
  (SKY130) reproduce every stored metric and the FoM exactly (largest
  difference 0). Re-running the calibration gave the archived values for
  PTM 65 nm and SKY130.
- **GF180MCU**: re-simulated designs do not match the logs, and not even each
  other, because of random transistor mismatch (Section 10). With
  `--gf180-pid`, the script redoes the GF180MCU calibration with ngspice
  started under the process IDs found for the archive, and reproduces
  `results/cal/gf180.json` exactly (all 16 numbers identical). This needs
  Linux and the `unshare` command (root, or unprivileged user namespaces).
  Example output:

  ```
  [ptm65]
    tr_CT1_ptm65_sc_s0: MATCH; FoM archived 3.792502, now 3.792502; largest relative metric difference 0
  ...
  [gf180]
    bo_CT1_gf180_s0: MISMATCH (expected for gf180: random mismatch, see --help); FoM archived 3.869404, now 3.136734; ...
    ...
    gf180 calibration (ngspice PIDs [10243, 10245, 10247, 10250, 10255, 10257, 10259, 10261]) MATCH with results/cal/gf180.json
  RESULT: all expected matches found
  ```
- The `count` modes of `chunk2.py` and `chunk3.py`, and `chunk.py`, report
  that no runs are missing once the archives are unpacked.

---

## 10. Notes on the calculations

- **Units.** Widths and lengths are in meters inside the code. For SKY130 the
  netlist uses `.option scale=1.0u`, so sizes are printed in micrometers.
  Capacitors are in farads and currents in amperes.
- **Unified FoM** (`env.py`). Power, noise, bandwidth, unity-gain frequency
  and transimpedance are first taken as log10. Each metric is then scaled to
  0-1 using the 2nd and 98th percentiles of the 300 random designs, capped at
  the 98th percentile plus 25% of the range, and clipped to -0.5 to 1.25.
  "Less is better" metrics (power, noise) enter as 1 minus the scaled value.
  The FoM is the sum over metrics, all with weight 1 (CT-1 has 5 metrics,
  the TIAs 4), and is never below 0. A metric that could not be measured
  counts as 0 ("more is better") or -0.5 ("less is better"). A CT-1 design
  with negative gain in dB, or a design with no measurable metric, gets FoM
  0.
- **Phase margin.** ngspice gives the phase in radians. The code converts it
  to degrees as 180 minus the phase drop from 10 Hz to the unity-gain
  frequency.
- **Noise.** `inoise_total` from ngspice's noise analysis. The caption in
  `gen_tables.py` states the unit as V^2 for CT-1 and A^2 for the TIAs.
- **Reward versus score.** The fuzzy reward only trains the agent. The logs
  and tables always use the unified FoM. The fuzzy system adapts toward the
  FoM divided by the number of metrics.
- **Calibration cache.** `surrogate.py` stores calibrations in the
  repository's `results/cal/` (or in the folder named by the environment
  variable `GCNSAC_CAL_DIR`), whatever folder a script is started from.
  Unpacking `results_logs.tar.gz` puts the archived files there. If a file
  is missing, the calibration is redone (8 ngspice runs) and saved there.
  The archived files for `sky130`, `ptm65` and `ptm45` have no EKV
  slope factor (`n_ekv`). For these, the EKV model uses the code's default
  1.3.
- **Output folder for tables.** `gen_tables.py` and `fig_advantage.py` write
  their LaTeX files to `results/tables/` and create it if needed.
- **Repeatability.** Every run uses fixed seeds, and PyTorch runs on one CPU
  thread. With the same model files, a re-run gives the same history (checked
  for PTM 65 nm). This holds for PTM and SKY130, not for GF180MCU (next
  point).
- **GF180MCU: every simulation includes random mismatch.** This is why
  GF180MCU numbers cannot be recomputed exactly. The model files and the
  ngspice version are not the cause. With the pinned files and ngspice 42, the
  archived calibration is reproduced bit for bit (below).
  - *The model files.* The GF180MCU file `design.ngspice` sets
    `sw_stat_mismatch = 1` (and `sw_stat_global = 1`) by default, and the
    code loads it unchanged. In the `typical` section, every `nmos_3p3` and
    `pmos_3p3` transistor is a subcircuit that adds a random threshold-voltage
    shift, `delvto = agauss(0, 0.7071 * par_vth * 1e-6 / sqrt(Leff * Weff), 1)`
    with `par_vth` = 0.007148 (NMOS) or 0.00666 (PMOS). For example, this gives a standard
    deviation of 3.7 mV for W = 4.4 um, L = 0.56 um. (The global process
    variation is not part of `typical`.) The SKY130 files loaded through the
    wrapper in `pdk_setup/` and the PTM cards contain no random functions, so
    these technologies are not affected.
  - *The random numbers.* ngspice 42, started with a circuit file
    (`ngspice -b file`, as the code does), seeds these random draws with its
    own process ID (PID). At start-up it calls `initw()`
    (`src/frontend/trannoise/wallace.c`). This runs `srand(getpid())` after
    `.spiceinit` has been read, so `set rndseed` or `setseed` there has no
    effect (I checked this). The code starts a new ngspice process for each
    simulation and does not record the PID. So each archived GF180MCU number is one
    random sample. The same design gives different results in different
    simulations. The archive itself shows this. For a given seed, `bo` and
    `mace` simulate the same 40 starting designs. Yet the first design of
    seed 1 on CT-1 scored 0 in `bo_CT1_gf180_s1` and 3.094 in
    `mace_CT1_gf180_s1`. Now take the six seed and circuit cases where at least one
    of the two logs has a non-zero FoM for that first design. In all six, the two values
    differ (CT-1 seeds 1 and 2, CT-2 seeds 2 and 3, CT-3 seeds 2 and 3).
  - *Size of the effect.* I re-simulated the best design of
    `bo_CT1_gf180_s0` (archived FoM 3.869) twenty times. The FoM ranged from 0 to 3.830
    (median 3.08), and three of the twenty scored 0. With mismatch switched off
    (`.param sw_stat_mismatch=0` after the `.lib` line), the same design
    gives 3.289, and the NMOS threshold of the
    calibration is 0.3614 V instead of the archived 0.3557 V. Because the
    best FoM of a run is the highest of many random samples, a re-simulation
    of that design usually scores lower than the logged value.
  - *What was reproduced.* I used the pinned GF180MCU files (Section 3.2) and
    ngspice 42. With the PIDs 10243, 10245, 10247 and 10250 (NMOS sweeps) and
    10255 (PMOS gate sweep), the calibration file
    `results/cal/gf180.json` is reproduced exactly. The three PMOS drain sweeps gave the lower
    limit 0.001 for every PID I tried. I found these PIDs by re-implementing the ngspice
    random-number steps (glibc `srand`/`rand`, the Tausworthe generator and
    `initw()`), and I checked them against real ngspice runs started under chosen PIDs
    in a new Linux PID namespace. Each ngspice 42 run uses two PID numbers
    (the process and one thread). This confirms the model files and the
    ngspice version. It needs a Linux system with glibc (I did not test other C libraries
    or ngspice versions).
  - *What cannot be reproduced.* The PIDs of the 128,100 GF180MCU
    simulations in the 218 GF180MCU run logs are unknown. So these logs
    cannot be recomputed exactly with any PDK version or setting. Switching mismatch
    off gives repeatable numbers, but they are different from the archived
    ones. I left the code unchanged, so that it still describes how the
    archived results were made.
  - *Calibration file date.* In `results_logs.tar.gz`, `results/cal/gf180.json`
    is dated 24 July 2026 06:13, later than most GF180MCU runs (from 23 July
    22:26), and it contains the EKV slope factor `n_ekv`, which the SKY130
    and PTM files written at 02:40 to 02:56 lack. No archived GF180MCU log
    counts the 8 calibration simulations (`sims_used` equals the budget in
    all of them). This suggests that GF180MCU runs of the graph-based agents
    finished before 06:13 used an earlier calibration, which is not in the
    archive. The file dates are the only evidence for this.
- **Which transfer runs the tables use.** `aggregate.py` uses `tr3_*_ft`
  (EKV-trained encoder) for "with transfer" when at least three such logs
  exist, otherwise `tr_*_ft`. "No transfer" is always `tr_*_sc`. Topology
  transfer uses `tt3_*_ft` when available and `tt_*_sc`. The `tr2_*`,
  `tt2_*` and `pia1500_*` logs are not used by any table or figure, and
  neither is `results/policy_ft/`. The topology panels of
  `F8a_tech_transfer.pdf` use `tt_*_ft`, not `tt3_*_ft`.
- **Weight files.** Only the graph-based agents save weights. Each `.pt` file
  holds the encoder and actor (and, for GCN-SAC, the two critics). The three
  `test_enconly_CT2_sky130_s*` weight files are not made by any script listed
  here and have no run log.
- **Temporary files.** Netlists are written to the system temporary folder
  and deleted after each simulation.

---

## 11. Version history

The repository has no tagged releases.

| Version | Date | Changes |
|---|---|---|
| Fixes (this version) | 30 Sep 2026 | Table and macro files go to `results/tables/`; calibration cache in `results/cal/`; campaign scripts no longer `cd /root/work/gcnsac` and create `results/`; `.gitignore`; PDK commits pinned; `check_pdk_reproduction.py`; GF180MCU reproducibility explained |
| Documentation update | 30 Sep 2026 | New README, CHANGELOG and CITATION; code and data unchanged |
| Initial upload | 24 Jul 2026 | Code, run-log archive, model archive and `pdk_setup/` |

You can find the details in [CHANGELOG.md](CHANGELOG.md).
