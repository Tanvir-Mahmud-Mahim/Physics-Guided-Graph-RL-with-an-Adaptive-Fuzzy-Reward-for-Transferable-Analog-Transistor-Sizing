# Physics-Guided Graph RL with an Adaptive Fuzzy Reward for Transferable Analog Transistor Sizing

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

This repository holds the code, trained models and run data for the paper:

**"Physics-Guided Graph Reinforcement Learning with an Adaptive Fuzzy Reward for Transferable Analog
Transistor Sizing"**

The paper has been submitted to IEEE TCAD (the IEEE Transactions on
Computer-Aided Design of Integrated Circuits and Systems).
The paper's author list is not recorded in this repository. I, Tanvir M. Mahim
(BRAC University), maintain this repository.

- Repository: https://github.com/Tanvir-Mahmud-Mahim/Physics-Guided-Graph-RL-with-an-Adaptive-Fuzzy-Reward-for-Transferable-Analog-Transistor-Sizing

Everything runs on open tools:

- the ngspice circuit simulator;
- the open GF180MCU and SKY130 process design kits (PDKs, the transistor model files a
  chip factory publishes);
- the ASU Predictive Technology Model (PTM) cards for 65 nm and 45 nm.

**No foundry PDK files are stored in this repository**.
Section 3 explains how you can download them.

According to the earlier README, the archived run logs are "the exact state
used to build every table and figure in the paper". When I rebuilt the summary
file and figures from them, they matched the archived copies exactly (see
[Section 9](DETAILS.md#9-built-in-checks)).

---

## Contents

1. [The idea in one minute](#1-the-idea-in-one-minute)
2. [What is in this repository](#2-what-is-in-this-repository)
3. [Installation](#3-installation)
4. [Quick start: three ways to use the code](#4-quick-start-three-ways-to-use-the-code)
5. [The scripts, step by step](DETAILS.md#5-the-scripts-step-by-step)
6. [Which script makes which figure and table](DETAILS.md#6-which-script-makes-which-figure-and-table)
7. [The Python modules](DETAILS.md#7-the-python-modules)
8. [Where the numbers come from](DETAILS.md#8-where-the-numbers-come-from)
9. [Built-in checks](DETAILS.md#9-built-in-checks)
10. [Notes on the calculations](DETAILS.md#10-notes-on-the-calculations)
11. [Version history](DETAILS.md#11-version-history)
12. [How to cite](#12-how-to-cite)
13. [License and contact](#13-license-and-contact)

Sections 5 to 11 are in [DETAILS.md](DETAILS.md), with a short guide in
[More details](#more-details-sections-5-to-11). DETAILS.md also has
[extra notes for Sections 1 to 4](DETAILS.md#extra-notes-for-sections-1-to-4).

---

## 1. The idea in one minute

An analog circuit such as an amplifier only works well if every transistor
has the right **size** (channel width W and length L). The bias current
and compensation capacitor must also be chosen well. Designers usually find these
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
  type-2 Takagi-Sugeno-Kang fuzzy system** (IT2-TSKF). It uses soft "low / medium /
  high" rules whose outputs adjust themselves during training.
- **Physics guidance.** A simple hand-written transistor model (square-law
  or the all-region EKV formula), fitted to the real simulator models, gives
  a direction in which the design should improve. The code calls this
  "PIA"; `surrogate.py` describes it as adjoint-guided policy learning.
- **Transfer.** An agent trained on the 180 nm GF180MCU process is reused
  on 130 nm (SKY130), 65 nm and 45 nm (PTM), or on a different circuit.

The code uses three test circuits: CT-1, a two-stage voltage amplifier; CT-2, a
two-stage transimpedance amplifier (TIA, it turns a current into a voltage);
and CT-3, a three-stage TIA. The code compares the method with:

- Bayesian optimization (BO), and a MACE-style multi-acquisition BO;
- A2C and PPO (standard RL without a graph);
- GCN-DDPG (the GCN-RL method of Wang et al., DAC 2020,
  as `gcnsac/agents.py` describes it).

Every method is scored with the **same
figure of merit** (FoM), which the simulation environment computes. So the fuzzy
reward only changes how an agent learns, never how it is scored.

---

## 2. What is in this repository

The main parts are:

- **gcnsac/**: the Python package. It holds the four technologies, the three test circuits,
  the ngspice runs and the FoM, the fuzzy reward, the physics model, the agents and the BO baselines.
- **scripts/**: one optimization run (`run.py`), the campaigns, the table and figure scripts,
  and `check_pdk_reproduction.py`.
- **pdk_setup/**: helper model files (no foundry PDK files).
- `results_logs.tar.gz`: the run logs, the summary file, calibration, normalization and figures.
- `models.tar.gz.part-*`: the trained network weights, split into parts.
- **README.md**, **CHANGELOG.md**, **CITATION.cff**, **LICENSE** and **.gitignore**.

The full file tree, and what `pdk_setup/` and the archives contain, are in
[DETAILS.md](DETAILS.md#more-on-section-2-the-files-and-archives).

The folders `results/` (after unpacking, and all generated outputs) and
`pdk/` (if you download the PDKs there) are not part of the repository. The
`.gitignore` file keeps them out of git.

---

## 3. Installation

### 3.1 Python packages and ngspice

The repository does not record a Python version. I checked the code
with **Python 3.11**. There is no `requirements.txt`, so install the packages directly:

```
pip install torch numpy scipy matplotlib
```

You also need the simulator **ngspice**, installed and callable as `ngspice`.
The earlier README names ngspice 42. For the checks here I used the Ubuntu package
ngspice 42+ds-3build1:

```
sudo apt install ngspice
```

Only the table and figure scripts (Way A in Section 4) work without ngspice
and without the PDKs.

### 3.2 Device models (PDKs)

The code looks for the model files in the folder named by the environment
variable `GCNSAC_PDK_ROOT` (default `~/work/pdk`). It expects these files:

| Technology | File(s) the code loads |
|---|---|
| `gf180` | `$GCNSAC_PDK_ROOT/gf180/models/ngspice/design.ngspice` and `sm141064.ngspice` (section `typical`) |
| `sky130` | `$GCNSAC_PDK_ROOT/sky130/models/mini_tt.spice` (the wrapper from `pdk_setup/`, which then loads the official files) |
| `ptm65` | `$GCNSAC_PDK_ROOT/ptm/65nm_bulk.pm` |
| `ptm45` | `$GCNSAC_PDK_ROOT/ptm/45nm_bulk.pm` |

Run this from the repository folder, on Linux or macOS:

```
export GCNSAC_PDK_ROOT=$PWD/pdk
mkdir -p pdk/ptm
cp pdk_setup/65nm_bulk.pm pdk_setup/45nm_bulk.pm pdk/ptm/

# GF180MCU (only models/ngspice is used), pinned to commit 9f992d5
git clone --filter=blob:none --no-checkout https://github.com/google/globalfoundries-pdk-libs-gf180mcu_fd_pr pdk/gf180
(cd pdk/gf180 && git sparse-checkout set models/ngspice && git checkout 9f992d5a9186d1f7820c58f039c484ad35b2edea)

# SKY130 (only the models and two transistor cells are used), pinned to commit f62031a
git clone --filter=blob:none --no-checkout https://github.com/google/skywater-pdk-libs-sky130_fd_pr pdk/sky130
(cd pdk/sky130 && git sparse-checkout set models cells/nfet_01v8 cells/pfet_01v8 && git checkout f62031a1be9aefe902d6d54cddd6f59b57627436)
cp pdk_setup/sky130_mini_tt.spice pdk/sky130/models/mini_tt.spice
```

The downloads took about 128 MB (GF180MCU) and 28 MB (SKY130) here. Both
upstream repositories use the Apache-2.0 license. `export` lasts only for the
current terminal, so set it again when you open a new one.

You can test your setup with `python3 scripts/check_pdk_reproduction.py` (Section 9).
**GF180MCU results cannot be reproduced exactly from these files alone.**
The GF180MCU model files switch on random transistor mismatch, and ngspice
seeds it with a number that the logs do not record (Section 10).

The package versions I used for the checks, and how I chose and checked the PDK
commits, are in [DETAILS.md](DETAILS.md#more-on-section-3-installation).

---

## 4. Quick start: three ways to use the code

Run all commands from the repository folder.

### Way A: rebuild the tables, text numbers and figures from the archived runs (about 15 seconds)

You need only Python for this (no ngspice, no PDKs).

```
tar xzf results_logs.tar.gz
cat models.tar.gz.part-* | tar xz
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
  `results/tables/` (they create the folder).
- The figures go to `results/figs/`.

You need both archives, because two figures read the `results/w/*_tskfhist.npy`
files from the model archive.

### Way B: recompute one archived run and compare (about 1 to 2 minutes)

You need only ngspice and the PTM cards for this (no PDK download). Run it after Way A (or at least
after `tar xzf results_logs.tar.gz`):

```
mkdir -p pdk/ptm
cp pdk_setup/65nm_bulk.pm pdk_setup/45nm_bulk.pm pdk/ptm/
export GCNSAC_PDK_ROOT=$PWD/pdk
python3 scripts/run.py --method gcnsac_tskf_pia --ct CT1 --pdk ptm65 --seed 0 --budget 150 --tag check_tr_CT1_ptm65_sc_s0
python3 -c "import json; a=json.load(open('results/tr_CT1_ptm65_sc_s0.json')); b=json.load(open('results/check_tr_CT1_ptm65_sc_s0.json')); print('same history:', a['history']==b['history'], a['best_fom'], b['best_fom'])"
```

This repeats the archived run `tr_CT1_ptm65_sc_s0` (the proposed method
trained from scratch on CT-1 at 65 nm, 150 simulations). When I ran it, it printed
`same history: True 3.792502335347418 3.792502335347418` (13 s, measured on
a shared 2-core machine). The run uses the archived calibration files in
`results/cal/` from `results_logs.tar.gz` (Section 10).

To check the model files of all four technologies at once, run
`python3 scripts/check_pdk_reproduction.py` (Section 9).

### Way C: recompute everything (long)

1. Install ngspice and all four technologies (Section 3).
2. Run the campaigns in this order, from the repository folder:
   `campaign_main.sh`, then `campaign2.sh` (or `resume_all.sh` for both),
   then `chunk2.py`, then `chunk3.py`. Details and caveats are in Section 5.
3. Then run the four commands of Way A (without unpacking the archives).

PTM and SKY130 runs repeat the archived logs exactly. GF180MCU runs do not.
Every GF180MCU simulation draws new random transistor mismatch (Section 10),
so recomputed GF180MCU numbers differ from the archived ones.

The archived logs record a total of 16.5 hours of single-run time for all
326 runs. The scripts run two jobs at a time. I did not re-run these campaigns
here.

---

## More details (Sections 5 to 11)

The full notes are in [DETAILS.md](DETAILS.md). Here is what each section holds:

- [5. The scripts, step by step](DETAILS.md#5-the-scripts-step-by-step):
  every command with its run time and outputs; the options of `run.py`; caveats.
- [6. Which script makes which figure and table](DETAILS.md#6-which-script-makes-which-figure-and-table):
  each output file and the script that makes it.
- [7. The Python modules](DETAILS.md#7-the-python-modules):
  one line on each file of the Python package.
- [8. Where the numbers come from](DETAILS.md#8-where-the-numbers-come-from):
  device models, size limits, circuit values, budgets, seeds and learning settings.
- [9. Built-in checks](DETAILS.md#9-built-in-checks):
  the checks I ran for this guide, with example output.
- [10. Notes on the calculations](DETAILS.md#10-notes-on-the-calculations):
  how the FoM is computed, repeatability, and why GF180MCU numbers cannot be recomputed exactly.
- [11. Version history](DETAILS.md#11-version-history):
  each version with its date and changes.

---

## 12. How to cite

GitHub shows a **"Cite this repository"** button in the right-hand column.
It reads `CITATION.cff`.

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
(Apache-2.0). I checked that this holds for the GF180MCU and SKY130 repositories you
download. The PTM cards in `pdk_setup/` come from the ASU
Predictive Technology Model. The repository does not state their license
terms.

If you have questions or find a bug, please open an issue on this repository. You can also
contact me, Tanvir M. Mahim, at BRAC University (tanvir.mahim@bracu.ac.bd).
