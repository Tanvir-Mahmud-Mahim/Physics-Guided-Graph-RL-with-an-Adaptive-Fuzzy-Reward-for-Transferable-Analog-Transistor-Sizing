#!/bin/bash
# Resume-aware campaign: runs every missing experiment (main, ablation,
# transfer), skipping any run whose results JSON already exists.
cd /root/work/gcnsac
LOG=results/campaign_resume.log

run_if_missing() {  # tag, command...
  local tag=$1; shift
  if [ ! -f "results/${tag}.json" ]; then
    echo "$*"
  fi
}

JOBS=()
# ---- main comparison ----
for ct in CT1 CT2 CT3; do
  for m in bo mace a2c ppo a2c_tskf ppo_tskf gcnddpg gcnsac gcnsac_tskf gcnsac_tskf_pia; do
    for s in 0 1 2; do
      tag="${m}_${ct}_gf180_s${s}"
      j=$(run_if_missing "$tag" "python3 scripts/run.py --method $m --ct $ct --seed $s --budget 600 --pdk gf180")
      [ -n "$j" ] && JOBS+=("$j")
    done
  done
done
printf "%s\n" "${JOBS[@]}" | xargs -P 2 -I {} sh -c '{} >> results/campaign_resume.log 2>&1'
echo "MAIN DONE" >> $LOG

# ---- norm stats for transfer nodes (sequential, no race) ----
for ct in CT1 CT2 CT3; do
  for pdk in sky130 ptm65 ptm45; do
    python3 - << EOF
import sys, os; sys.path.insert(0, ".")
from gcnsac.env import CircuitEnv
p = "results/norm/${ct}_${pdk}.json"
if not os.path.exists(p):
    CircuitEnv("$ct", "$pdk").calibrate(n_samples=300, seed=1234, save=p)
EOF
  done
done

JOBS=()
# ---- ablations ----
for m in gcnsac_tskf_pia_noper gcnsac_tskf_pia_noaug gcnsac_tskf_pia_nosimclr gcnsac_tskf_pia_t1; do
  for s in 0 1 2; do
    tag="${m}_CT2_gf180_s${s}"
    j=$(run_if_missing "$tag" "python3 scripts/run.py --method $m --ct CT2 --seed $s --budget 600 --pdk gf180")
    [ -n "$j" ] && JOBS+=("$j")
  done
done
# ---- technology transfer ----
for ct in CT1 CT2 CT3; do
  for pdk in sky130 ptm65 ptm45; do
    for s in 0 1 2; do
      t1="tr_${ct}_${pdk}_ft_s${s}"
      j=$(run_if_missing "$t1" "python3 scripts/run.py --method gcnsac_tskf_pia --ct $ct --pdk $pdk --seed $s --budget 150 --load results/w/gcnsac_tskf_pia_${ct}_gf180_s${s}.pt --tag $t1")
      [ -n "$j" ] && JOBS+=("$j")
      t2="tr_${ct}_${pdk}_sc_s${s}"
      j=$(run_if_missing "$t2" "python3 scripts/run.py --method gcnsac_tskf_pia --ct $ct --pdk $pdk --seed $s --budget 150 --tag $t2")
      [ -n "$j" ] && JOBS+=("$j")
    done
  done
done
# ---- topology transfer ----
for pair in "CT2 CT3" "CT3 CT2"; do
  set -- $pair
  src=$1; dst=$2
  for s in 0 1 2; do
    t1="tt_${src}to${dst}_ft_s${s}"
    j=$(run_if_missing "$t1" "python3 scripts/run.py --method gcnsac_tskf_pia --ct $dst --pdk gf180 --seed $s --budget 150 --load results/w/gcnsac_tskf_pia_${src}_gf180_s${s}.pt --tag $t1")
    [ -n "$j" ] && JOBS+=("$j")
    t2="tt_${dst}_sc_s${s}"
    j=$(run_if_missing "$t2" "python3 scripts/run.py --method gcnsac_tskf_pia --ct $dst --pdk gf180 --seed $s --budget 150 --tag $t2")
    [ -n "$j" ] && JOBS+=("$j")
  done
done
printf "%s\n" "${JOBS[@]}" | xargs -P 2 -I {} sh -c '{} >> results/campaign_resume.log 2>&1'
echo "ALL DONE" >> $LOG
