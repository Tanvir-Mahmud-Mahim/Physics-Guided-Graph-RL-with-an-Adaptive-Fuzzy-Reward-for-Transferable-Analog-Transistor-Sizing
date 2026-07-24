#!/bin/bash
# Phase 2: waits for the main campaign, then runs ablations + transfer studies.
cd /root/work/gcnsac
while ! grep -q "MAIN CAMPAIGN DONE" results/campaign_main.log 2>/dev/null; do
  sleep 60
done

# ---- pre-generate normalization stats sequentially (no race) ----
for ct in CT1 CT2 CT3; do
  for pdk in sky130 ptm65 ptm45; do
    python3 - << EOF
import sys, os; sys.path.insert(0, ".")
from gcnsac.env import CircuitEnv
env = CircuitEnv("$ct", "$pdk")
p = "results/norm/${ct}_${pdk}.json"
if not os.path.exists(p):
    env.calibrate(n_samples=300, seed=1234, save=p)
EOF
  done
done

JOBS=()
# ---- ablations on CT2 (gf180, 600 steps) ----
for m in gcnsac_tskf_pia_noper gcnsac_tskf_pia_noaug gcnsac_tskf_pia_nosimclr gcnsac_tskf_pia_t1; do
  for s in 0 1 2; do
    JOBS+=("python3 scripts/run.py --method $m --ct CT2 --seed $s --budget 600 --pdk gf180")
  done
done
# ---- technology transfer: 180nm pretrain -> {130,65,45} nm ----
for ct in CT1 CT2 CT3; do
  for pdk in sky130 ptm65 ptm45; do
    for s in 0 1 2; do
      JOBS+=("python3 scripts/run.py --method gcnsac_tskf_pia --ct $ct --pdk $pdk --seed $s --budget 150 --load results/w/gcnsac_tskf_pia_${ct}_gf180_s${s}.pt --tag tr_${ct}_${pdk}_ft_s${s}")
      JOBS+=("python3 scripts/run.py --method gcnsac_tskf_pia --ct $ct --pdk $pdk --seed $s --budget 150 --tag tr_${ct}_${pdk}_sc_s${s}")
    done
  done
done
# ---- topology transfer on gf180 (encoder reuse) ----
for pair in "CT2 CT3" "CT3 CT2"; do
  set -- $pair
  src=$1; dst=$2
  for s in 0 1 2; do
    JOBS+=("python3 scripts/run.py --method gcnsac_tskf_pia --ct $dst --pdk gf180 --seed $s --budget 150 --load results/w/gcnsac_tskf_pia_${src}_gf180_s${s}.pt --tag tt_${src}to${dst}_ft_s${s}")
    JOBS+=("python3 scripts/run.py --method gcnsac_tskf_pia --ct $dst --pdk gf180 --seed $s --budget 150 --tag tt_${dst}_sc_s${s}")
  done
done

printf "%s\n" "${JOBS[@]}" | xargs -P 2 -I {} sh -c '{} >> results/campaign2.log 2>&1'
echo "PHASE2 DONE" >> results/campaign2.log
