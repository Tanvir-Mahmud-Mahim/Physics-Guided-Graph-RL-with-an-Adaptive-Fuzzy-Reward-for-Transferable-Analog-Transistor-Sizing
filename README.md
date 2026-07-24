# Physics-Guided Graph RL with an Adaptive Fuzzy Reward for Transferable Analog Transistor Sizing

Open-source codebase, trained models, and run data for the paper
"Physics-Guided Graph Reinforcement Learning with an Adaptive Fuzzy Reward
for Transferable Analog Transistor Sizing" (submitted to IEEE TCAD).

Everything runs on open tools: ngspice 42, the GF180MCU and SKY130 open
PDKs, and ASU PTM predictive model cards for 65 nm and 45 nm. No proprietary
or NDA content is required.

## Layout

```
gcnsac/
  pdk.py        PDK abstraction (GF180MCU, SKY130, PTM65, PTM45)
  circuits.py   parameterized netlists + circuit graphs (CT-1, CT-2, CT-3)
  env.py        ngspice-in-the-loop environment, unified environment FoM
  tskf.py       interval type-2 TSKF reward (KM type reduction, adaptation)
  surrogate.py  PDK-calibrated differentiable physics surrogate (adjoint)
  agents.py     GCN-SAC (+PER, augmentation, SimCLR), GCN-DDPG, A2C, PPO
  baselines.py  BO (GP-EI) and MACE-style multi-acquisition BO
scripts/
  run.py            single experiment runner
  campaign_main.sh  main comparison (11 methods x 3 circuits x 5 seeds)
  campaign2.sh      ablations + technology/topology transfer
  chunk3.py         EKV flagship campaign runner (resumable worker mode)
  aggregate.py      collect all runs into results/agg/aggregate.json
  gen_tables.py     LaTeX tables + text macros (all paper numbers)
  figures.py        all paper figures
  fig_advantage.py  Fig. 3 (final quality + budget friendliness)
results/            run logs (JSON), weights, normalization stats, figures
```

## Trained models and run data

To keep the repository within web-upload size limits, the run data and the
trained model weights are stored as compressed archives:

- `results_logs.tar.gz` holds every run log (JSON), the aggregate file,
  the normalization statistics, the calibration constants, and the paper
  figures. Extract at the repository root:
  `tar xzf results_logs.tar.gz`
- `models.tar.gz.part-00` ... `part-09` hold all 448 trained model
  checkpoints (`results/w/*.pt`, PyTorch). Rejoin and extract at the
  repository root:
  `cat models.tar.gz.part-* | tar xz`

After both extractions, `results/` matches the exact state used to build
every table and figure in the paper:
`python3 scripts/aggregate.py && python3 scripts/gen_tables.py && python3 scripts/figures.py`.

## Setup

```bash
sudo apt install ngspice
pip install torch numpy scipy matplotlib
# PDKs (see gcnsac/pdk.py for expected paths, override with GCNSAC_PDK_ROOT):
git clone --depth 1 https://github.com/google/globalfoundries-pdk-libs-gf180mcu_fd_pr pdk/gf180
git clone --depth 1 --filter=blob:none --sparse https://github.com/google/skywater-pdk-libs-sky130_fd_pr pdk/sky130
(cd pdk/sky130 && git sparse-checkout set models cells/nfet_01v8 cells/pfet_01v8)
# copy pdk/sky130_mini_tt.spice (provided) to pdk/sky130/models/mini_tt.spice
# PTM 65/45 nm bulk model cards go to pdk/ptm/{65nm_bulk.pm,45nm_bulk.pm}
```

## Reproduce the paper

```bash
bash scripts/campaign_main.sh     # main comparison
bash scripts/campaign2.sh         # ablations + transfer
python3 scripts/aggregate.py
python3 scripts/gen_tables.py
python3 scripts/figures.py
```

Budgets: 600 simulator evaluations per optimization run, 300 normalization
samples per circuit/node (shared), 24 calibration DC sweeps per node
(shared), 150 evaluations for fine-tuning runs. Seeds 0-4 (five seeds).

## License

Apache-2.0 (see `LICENSE`). PDK files retain their upstream licenses
(Apache-2.0).
