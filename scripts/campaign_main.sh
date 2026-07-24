#!/bin/bash
# Main comparison campaign: 10 methods x 3 CTs x 3 seeds, 600-step budget, GF180MCU
cd /root/work/gcnsac
JOBS=()
for ct in CT1 CT2 CT3; do
  for m in bo mace a2c ppo a2c_tskf ppo_tskf gcnddpg gcnsac gcnsac_tskf gcnsac_tskf_pia; do
    for s in 0 1 2; do
      JOBS+=("python3 scripts/run.py --method $m --ct $ct --seed $s --budget 600 --pdk gf180")
    done
  done
done
printf "%s\n" "${JOBS[@]}" | xargs -P 2 -I {} sh -c '{} >> results/campaign_main.log 2>&1'
echo "MAIN CAMPAIGN DONE" >> results/campaign_main.log
