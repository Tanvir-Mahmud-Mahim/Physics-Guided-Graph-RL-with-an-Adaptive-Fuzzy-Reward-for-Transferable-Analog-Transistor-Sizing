# Physics-Guided Graph RL with an Adaptive Fuzzy Reward for Transferable Analog Transistor Sizing

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

Code, trained models and run data for the paper **"Physics-Guided Graph
Reinforcement Learning with an Adaptive Fuzzy Reward for Transferable Analog
Transistor Sizing"** (submitted to IEEE TCAD, the IEEE Transactions on
Computer-Aided Design of Integrated Circuits and Systems).
The paper's author list is not recorded in this repository; the repository is
maintained by Tanvir M. Mahim (BRAC University).

- Repository: https://github.com/Tanvir-Mahmud-Mahim/Physics-Guided-Graph-RL-with-an-Adaptive-Fuzzy-Reward-for-Transferable-Analog-Transistor-Sizing

Everything runs on open tools: the ngspice circuit simulator, the open
GF180MCU and SKY130 process design kits (PDKs, the transistor model files a
chip factory publishes), and the ASU Predictive Technology Model (PTM) cards
for 65 nm and 45 nm. **No foundry PDK files are stored in this repository**;
Section 3 explains how to download them.

According to the earlier README, the archived run logs are "the exact state
used to build every table and figure in the paper". Rebuilding the summary
file and figures from them reproduces the archived copies exactly (checked
here; see [Section 9](#9-built-in-checks)).

---

## Contents

1. [The idea in one minute](#1-the-idea-in-one-minute)
2. [What is in this repository](#2-what-is-in-this-repository)
3. [Installation](#3-installation)
4. [Quick start: three ways to use the code](#4-quick-start-three-ways-to-use-the-code)
5. [The scripts, step by step](#5-the-scripts-step-by-step)
6. [Which script makes which figure and table](#6-which-script-makes-which-figure-and-table)
7. [The Python modules](#7-the-python-modules)
8. [Where the numbers come from](#8-where-the-numbers-come-from)
9. [Built-in checks](#9-built-in-checks)
10. [Notes on the calculations](#10-notes-on-the-calculations)
11. [Version history](#11-version-history)
12. [How to cite](#12-how-to-cite)
13. [License and contact](#13-license-and-contact)

---

## 1. The idea in one minute

An analog circuit such as an amplifier only works well if every transistor
has the right **size** (channel width W and length L), and the bias current
and compensation capacitor are chosen well. Designers usually find these
values by trial and error with a circuit simulator. Each trial costs one
simulation, so a good method must find a good design in few simulations.

This code lets a computer learn the sizing by trial and reward
(**reinforcement learning**, RL). Its main parts are:

- **The circuit as a graph.** Each device (transistor, capacitor, current
  source) is a point, and devices that are wired together are linked. A
  **graph convolutional network** (GCN, a neural network that works on such
  graphs) reads this picture. The learning method is **soft actor-critic**
  (SAC), so the agent is called GCN-SAC.
- **An adaptive fuzzy reward.** The measured performance (gain, bandwidth,
  power, noise, ...) is turned into a training reward by an **interval
  type-2 Takagi-Sugeno-Kang fuzzy system** (IT2-TSKF): soft "low / medium /
  high" rules whose outputs adjust themselves during training.
- **Physics guidance.** A simple hand-written transistor model (square-law
  or the all-region EKV formula), fitted to the real simulator models, gives
  a direction in which the design should improve. The code calls this
  "PIA"; `surrogate.py` describes it as adjoint-guided policy learning.
- **Transfer.** An agent trained on the 180 nm GF180MCU process is reused
  on 130 nm (SKY130), 65 nm and 45 nm (PTM), or on a different circuit.

Three test circuits are used: CT-1, a two-stage voltage amplifier; CT-2, a
two-stage transimpedance amplifier (TIA, it turns a current into a voltage);
and CT-3, a three-stage TIA. The code compares the method with Bayesian
optimization (BO), a MACE-style multi-acquisition BO, A2C and PPO (standard
RL without a graph), and GCN-DDPG (the GCN-RL method of Wang et al., DAC 2020,
as `gcnsac/agents.py` describes it). Every method is scored with the **same
figure of merit** (FoM), computed by the simulation environment, so the fuzzy
reward only changes how an agent learns, never how it is scored.

---

## 2. What is in this repository

```
Physics-Guided-Graph-RL-...-Transistor-Sizing/
|-- README.md              this guide
|-- CHANGELOG.md           what changed, newest first
|-- CITATION.cff           citation details (drives the "Cite this repository" button)
|-- LICENSE                Apache-2.0 license
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
|   |-- gen_tables.py      LaTeX tables and text macros for the paper
|   |-- figures.py         paper figures (PDF)
|   `-- fig_advantage.py   the "advantage" figure and its text macros
|-- pdk_setup/             helper model files (no foundry PDK files, see below)
|   |-- 65nm_bulk.pm       PTM 65 nm model card (text)
|   |-- 45nm_bulk.pm       PTM 45 nm metal-gate/high-k model card (text)
|   `-- sky130_mini_tt.spice  23-line wrapper that loads the SKY130 typical models
|-- results_logs.tar.gz    run logs, aggregate file, calibration, normalization, figures (0.7 MB)
`-- models.tar.gz.part-00 ... part-09   trained network weights, split into 10 parts (86 MB packed, 95 MB unpacked)
```

**What `pdk_setup/` contains.** Two PTM model cards (`65nm_bulk.pm`, 146
lines, with the comments "PTM 65nm NMOS" and "PTM 65nm PMOS"; `45nm_bulk.pm`,
141 lines, headed "PTM High Performance 45nm Metal Gate / High-K /
Strained-Si") and a small wrapper,
`sky130_mini_tt.spice`. The wrapper holds no model data: it includes four
files from the official SKY130 repository (`parameters/invariant.spice`,
`parameters/lod.spice`, and the typical-corner `nfet_01v8` and `pfet_01v8`
model files) and sets 18 parameters to zero (9 per transistor type, with
names ending in `_slope`, `_slope1` or `_slope_spectre`).

**What the archives contain** (listed with `tar tzvf`; nothing else is inside):

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
checkpoints (`results/w/*.pt`)"; the 448 files are in fact 239 `.pt` files
plus 209 `.npy` files. Neither archive contains PDK or model-card files.

The folders `results/` (after unpacking) and `pdk/` (if you download the
PDKs there) are not part of the repository. There is no `.gitignore`, so git
lists them as untracked; please do not commit them.

---

## 3. Installation

### 3.1 Python packages and ngspice

The Python version is not recorded in the repository; the code was checked
here with **Python 3.11**. There is no `requirements.txt`. Install:

```
pip install torch numpy scipy matplotlib
```

Versions used for the checks in this guide: Python 3.11.15, PyTorch 2.14.0
(only the CPU is used), NumPy 2.4.4, SciPy 1.17.1, Matplotlib 3.10.9.

The simulator **ngspice** must be installed and callable as `ngspice`
(the earlier README names ngspice 42; the checks here used the Ubuntu package
ngspice 42+ds-3build1):

```
sudo apt install ngspice
```

Only the table and figure scripts (Way A in Section 4) work without ngspice
and without the PDKs.

### 3.2 Device models (PDKs)

The code looks for the model files in the folder named by the environment
variable `GCNSAC_PDK_ROOT` (default `~/work/pdk`). It expects:

| Technology | File(s) the code loads |
|---|---|
| `gf180` | `$GCNSAC_PDK_ROOT/gf180/models/ngspice/design.ngspice` and `sm141064.ngspice` (section `typical`) |
| `sky130` | `$GCNSAC_PDK_ROOT/sky130/models/mini_tt.spice` (the wrapper from `pdk_setup/`, which then loads the official files) |
| `ptm65` | `$GCNSAC_PDK_ROOT/ptm/65nm_bulk.pm` |
| `ptm45` | `$GCNSAC_PDK_ROOT/ptm/45nm_bulk.pm` |

From the repository folder, on Linux or macOS:

```
export GCNSAC_PDK_ROOT=$PWD/pdk
mkdir -p pdk/ptm
cp pdk_setup/65nm_bulk.pm pdk_setup/45nm_bulk.pm pdk/ptm/

# GF180MCU (only models/ngspice is used)
git clone --depth 1 --filter=blob:none --sparse https://github.com/google/globalfoundries-pdk-libs-gf180mcu_fd_pr pdk/gf180
(cd pdk/gf180 && git sparse-checkout set models/ngspice)

# SKY130 (only the models and two transistor cells are used)
git clone --depth 1 --filter=blob:none --sparse https://github.com/google/skywater-pdk-libs-sky130_fd_pr pdk/sky130
(cd pdk/sky130 && git sparse-checkout set models cells/nfet_01v8 cells/pfet_01v8)
cp pdk_setup/sky130_mini_tt.spice pdk/sky130/models/mini_tt.spice
```

A full `git clone --depth 1` of the GF180MCU repository (as in the earlier
README) gives the same files. The sparse downloads took about 128 MB
(GF180MCU) and 28 MB (SKY130) here. Both upstream repositories are licensed
Apache-2.0. `export` lasts only for the current terminal; set it again in a
new one.

**Which PDK version?** The repository does not record the versions used for
the paper. The checks here used the upstream `main` branch: GF180MCU commit
`9f992d5` (31 May 2023) and SKY130 commit `f62031a`. With these, SKY130 and
PTM results matched the archived logs exactly, but GF180MCU results did not
(see [Section 10](#10-notes-on-the-calculations)).

---

## 4. Quick start: three ways to use the code

Run all commands from the repository folder.

### Way A: rebuild the tables, text numbers and figures from the archived runs (about 15 seconds)

Needs Python only (no ngspice, no PDKs).

```
tar xzf results_logs.tar.gz
cat models.tar.gz.part-* | tar xz
mkdir -p ~/work/paper
cp results/agg/aggregate.json aggregate_archived.json
python3 scripts/aggregate.py
cmp results/agg/aggregate.json aggregate_archived.json && echo "aggregate.json unchanged"
python3 scripts/gen_tables.py
python3 scripts/figures.py
python3 scripts/fig_advantage.py
```

- `aggregate.py` prints the mean and spread of the FoM for each method and
  circuit, and rewrites `results/agg/aggregate.json` (byte-identical to the
  archived copy).
- `gen_tables.py` and `fig_advantage.py` write LaTeX files into
  `~/work/paper/`. **This folder must exist** (hence the `mkdir`); otherwise
  they stop with `FileNotFoundError`.
- The figures go to `results/figs/`.

Both archives are needed: two figures read the `results/w/*_tskfhist.npy`
files from the model archive.

### Way B: recompute one archived run and compare (about 1 to 2 minutes)

Needs ngspice and the PTM cards only (no PDK download). After Way A (or at least
after `tar xzf results_logs.tar.gz`):

```
mkdir -p pdk/ptm
cp pdk_setup/65nm_bulk.pm pdk_setup/45nm_bulk.pm pdk/ptm/
export GCNSAC_PDK_ROOT=$PWD/pdk
mkdir -p ~/work/gcnsac/results/cal
cp results/cal/*.json ~/work/gcnsac/results/cal/
python3 scripts/run.py --method gcnsac_tskf_pia --ct CT1 --pdk ptm65 --seed 0 --budget 150 --tag check_tr_CT1_ptm65_sc_s0
python3 -c "import json; a=json.load(open('results/tr_CT1_ptm65_sc_s0.json')); b=json.load(open('results/check_tr_CT1_ptm65_sc_s0.json')); print('same history:', a['history']==b['history'], a['best_fom'], b['best_fom'])"
```

This repeats the archived run `tr_CT1_ptm65_sc_s0` (the proposed method
trained from scratch on CT-1 at 65 nm, 150 simulations). Here it printed
`same history: True 3.792502335347418 3.792502335347418`. The copy into
`~/work/gcnsac/results/cal/` is explained in Section 10 (it overwrites any
calibration files you already have there).

### Way C: recompute everything (long)

1. Install ngspice and all four technologies (Section 3).
2. `mkdir -p results` (the campaign scripts write their logs there; if the
   folder is missing, no run starts).
3. Run the campaigns in this order, from the repository folder:
   `campaign_main.sh`, then `campaign2.sh` (or `resume_all.sh` for both),
   then `chunk2.py`, then `chunk3.py`. Details and caveats are in Section 5.
4. Then run the four commands of Way A (without unpacking the archives).

The archived logs record a total of 16.5 hours of single-run time for all
326 runs; the scripts run two jobs at a time. These campaigns were not re-run
here.

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
| 7 | `python3 scripts/gen_tables.py` | Writes the paper's LaTeX tables and text macros (needs `~/work/paper/`) | 2.6 s | `~/work/paper/*.tex` |
| 8 | `python3 scripts/figures.py` | Draws the paper figures | 8.4 s | `results/figs/*.pdf` |
| 9 | `python3 scripts/fig_advantage.py` | Draws the advantage figure and writes its macros (needs `~/work/paper/`) | 1.9 s | `results/figs/F_advantage.pdf`, `~/work/paper/adv_macros.tex` |

\*Times measured on a shared two-core computer, except "logs", which is the
sum of the `wall_s` field over the archived runs of that step (single-run
times on the author's machine).

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

- `campaign_main.sh`, `campaign2.sh` and `resume_all.sh` begin with
  `cd /root/work/gcnsac` (the author's folder). Elsewhere this line prints an
  error and the script carries on in the current folder, so start them from
  the repository folder.
- These three scripts send each run's output to `results/<name>.log`. If
  `results/` does not exist yet, no run starts; create it first
  (`mkdir -p results`).
- `chunk.py`, `chunk2.py`, `chunk3.py` and `worker.py` use relative paths
  and must also be started from the repository folder.
- If a normalization file is missing, `run.py` first simulates 300 random
  designs to make it (`norm_sims` in the log). All archived logs have
  `norm_sims` = 0, because the files already existed.

---

## 6. Which script makes which figure and table

The paper's figure numbers are not recorded in the repository, except that
the earlier README called `F_advantage.pdf` "Fig. 3". The table lists the
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
| `~/work/paper/tab_fom.tex` | Main FoM table (mean and spread, 11 methods, 3 circuits) | `aggregate.json` | `gen_tables.py` |
| `~/work/paper/tab_metrics.tex` | Measured gain, bandwidth, power and noise of each method's best design | `aggregate.json` | `gen_tables.py` |
| `~/work/paper/tab_transfer.tex`, `tab_transfer_full.tex` | Transfer results (fine-tuned versus from scratch; the full version adds topology transfer) | `aggregate.json` | `gen_tables.py` |
| `~/work/paper/tab_perseed.tex` | Per-seed FoM and Welch t-test against the three strongest other methods | run logs | `gen_tables.py` |
| `~/work/paper/tab_cost.tex` | Simulations and mean run time per method | `aggregate.json` | `gen_tables.py` |
| `~/work/paper/results_macros.tex`, `adv_macros.tex` | Numbers and sentences used in the paper text | `aggregate.json`, run logs | `gen_tables.py`, `fig_advantage.py` |

The regenerated figures were compared with the archived PDFs by rendering
both to images: all seven that `figures.py` and `fig_advantage.py` redraw
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
50 uA); CT-2 has 10 and CT-3 has 13 (input device W and L, then W of the
NMOS and PMOS and a shared L per stage, the capacitor 0.05 to 5 pF, the bias
current 1 to 100 uA). Widths range from twice the smallest width of the node
to the largest, except the TIA input device (smallest width to 0.3 x the
largest); lengths use the node's full L range. All values are set on a
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
gate-voltage sweeps and 6 drain-voltage sweeps); it is stored and reused, and
it is counted in `sims_used` only in the run that made it (these runs show
158 instead of 150). The earlier README and `gen_tables.py` say "24
calibration DC sweeps"; the code performs 8.

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

The repository has no automated test suite. These checks were run for this
guide:

- **Tables and figures from the archived logs** (Way A): `aggregate.py`
  rebuilds `results/agg/aggregate.json` byte for byte; the seven figures that
  `figures.py` and `fig_advantage.py` redraw are pixel-identical to the
  archived PDFs.
- **One run from scratch** (Way B): `tr_CT1_ptm65_sc_s0` was re-run and gave
  the identical history, best FoM and best design.
- **Re-simulating stored designs**: the best designs of
  `tr_CT1_ptm65_sc_s0` (PTM 65 nm) and `tr_CT2_sky130_sc_s1` (SKY130)
  reproduced their stored metrics exactly. Re-running the calibration gave the
  archived values for PTM 65 nm and SKY130.
- **GF180MCU did not match**: see Section 10.
- The `count` modes of `chunk2.py` and `chunk3.py`, and `chunk.py`, report
  that no runs are missing once the archives are unpacked.

---

## 10. Notes on the calculations

- **Units.** Widths and lengths are in metres inside the code; for SKY130 the
  netlist uses `.option scale=1.0u`, so sizes are printed in micrometres.
  Capacitors are in farads and currents in amperes.
- **Unified FoM** (`env.py`). Power, noise, bandwidth, unity-gain frequency
  and transimpedance are first taken as log10. Each metric is then scaled to
  0-1 using the 2nd and 98th percentiles of the 300 random designs, capped at
  the 98th percentile plus 25% of the range, and clipped to -0.5 to 1.25.
  "Less is better" metrics (power, noise) enter as 1 minus the scaled value.
  The FoM is the sum over metrics, all with weight 1 (CT-1 has 5 metrics,
  the TIAs 4), and is never below 0. A metric that could not be measured
  counts as 0 ("more is better") or -0.5 ("less is better"); a CT-1 design
  with negative gain in dB, or a design with no measurable metric, gets FoM
  0.
- **Phase margin.** ngspice gives the phase in radians; the code converts it
  to degrees as 180 minus the phase drop from 10 Hz to the unity-gain
  frequency.
- **Noise.** `inoise_total` from ngspice's noise analysis. The caption in
  `gen_tables.py` states the unit as V^2 for CT-1 and A^2 for the TIAs.
- **Reward versus score.** The fuzzy reward only trains the agent; the logs
  and tables always use the unified FoM. The fuzzy system adapts toward the
  FoM divided by the number of metrics.
- **Calibration cache.** `surrogate.py` stores calibrations in
  `~/work/gcnsac/results/cal/`, a fixed folder in your home directory, not in
  the repository's `results/cal/`. If a file is missing there, the
  calibration is redone (8 ngspice runs). Way B copies the archived files
  there. The archived files for `sky130`, `ptm65` and `ptm45` have no EKV
  slope factor (`n_ekv`); for these, the EKV model uses the code's default
  1.3.
- **Output folder for tables.** `gen_tables.py` and `fig_advantage.py` write
  to `~/work/paper/`, which must exist.
- **Repeatability.** Every run uses fixed seeds, and PyTorch runs on one CPU
  thread. With the same model files, a re-run gives the same history (checked
  for PTM 65 nm).
- **GF180MCU model version.** With the current upstream GF180MCU files
  (commit `9f992d5`), re-simulating stored GF180MCU designs gave different
  metrics from the logs (for example, FoM 2.76 instead of 3.87 for the best
  design of `bo_CT1_gf180_s0`), a re-run of `a2c_CT3_gf180_s1` differed from
  the first simulation on, and the calibration differed slightly (NMOS
  threshold 0.360 V instead of 0.356 V). The exact GF180MCU files and ngspice
  settings used for the paper are not recorded, so recomputed GF180MCU
  numbers can differ from the archived ones.
- **Which transfer runs the tables use.** `aggregate.py` uses `tr3_*_ft`
  (EKV-trained encoder) for "with transfer" when at least three such logs
  exist, otherwise `tr_*_ft`; "no transfer" is always `tr_*_sc`. Topology
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
| Documentation update (this version) | 30 Sep 2026 | New README, CHANGELOG and CITATION; code and data unchanged |
| Initial upload | 24 Jul 2026 | Code, run-log archive, model archive and `pdk_setup/` |

Details are in [CHANGELOG.md](CHANGELOG.md).

---

## 12. How to cite

GitHub shows a **"Cite this repository"** button in the right-hand column,
which reads `CITATION.cff`.

> T. M. Mahim, "Physics-Guided Graph RL with an Adaptive Fuzzy Reward for
> Transferable Analog Transistor Sizing" (code, trained models and run data),
> GitHub (2026),
> https://github.com/Tanvir-Mahmud-Mahim/Physics-Guided-Graph-RL-with-an-Adaptive-Fuzzy-Reward-for-Transferable-Analog-Transistor-Sizing

The paper, "Physics-Guided Graph Reinforcement Learning with an Adaptive Fuzzy
Reward for Transferable Analog Transistor Sizing", has been submitted to IEEE
TCAD. Please cite it once it is published.

---

## 13. License and contact

The repository is released under the Apache License 2.0 (see `LICENSE`).
The earlier README adds that PDK files keep their upstream licenses
(Apache-2.0); this holds for the GF180MCU and SKY130 repositories you
download (checked). The PTM cards in `pdk_setup/` come from the ASU
Predictive Technology Model; the repository does not state their license
terms.

Questions and bug reports: please open an issue on this repository, or
contact Tanvir M. Mahim, BRAC University (tanvir.mahim@bracu.ac.bd).
