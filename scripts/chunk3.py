#!/usr/bin/env python3
"""EKV-flagship campaign: the all-region EKV surrogate configuration on all
circuits (5 seeds), plus transfer rerun from EKV-pretrained encoders."""
import os, subprocess, sys, time

RES = "results"

def missing(tag):
    return not os.path.exists(os.path.join(RES, tag + ".json"))

def ready(cmd):
    if "--load " in cmd:
        return os.path.exists(cmd.split("--load ")[1].split()[0])
    return True

def all_jobs():
    jobs = []
    for ct in ("CT1", "CT3"):          # CT2 already done (5 seeds)
        for s in range(5):
            tag = f"gcnsac_tskf_pia_ekv_{ct}_gf180_s{s}"
            jobs.append((tag, f"python3 scripts/run.py --method "
                         f"gcnsac_tskf_pia_ekv --ct {ct} --seed {s} "
                         f"--budget 600 --pdk gf180", 400))
    for ct in ("CT1", "CT2", "CT3"):
        for pdk in ("sky130", "ptm65", "ptm45"):
            for s in (0, 1, 2):
                tag = f"tr3_{ct}_{pdk}_ft_s{s}"
                jobs.append((tag, f"python3 scripts/run.py --method "
                             f"gcnsac_tskf_pia_ekv --ct {ct} --pdk {pdk} "
                             f"--seed {s} --budget 150 "
                             f"--load results/w/gcnsac_tskf_pia_ekv_{ct}_gf180_s{s}.pt "
                             f"--enc_only --tag {tag}", 150))
    for src, dst in (("CT2", "CT3"), ("CT3", "CT2")):
        for s in (0, 1, 2):
            tag = f"tt3_{src}to{dst}_ft_s{s}"
            jobs.append((tag, f"python3 scripts/run.py --method "
                         f"gcnsac_tskf_pia_ekv --ct {dst} --pdk gf180 "
                         f"--seed {s} --budget 150 "
                         f"--load results/w/gcnsac_tskf_pia_ekv_{src}_gf180_s{s}.pt "
                         f"--enc_only --tag {tag}", 150))
    return jobs

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "worker"
    jobs = [(t, c, e) for (t, c, e) in all_jobs() if missing(t) and ready(c)]
    if mode == "count":
        blocked = sum(1 for (t, c, e) in all_jobs() if missing(t) and not ready(c))
        print(len(jobs), "ready,", blocked, "blocked")
        return
    # worker mode: loop with claim files
    CLAIM = os.path.join(RES, "claims3")
    os.makedirs(CLAIM, exist_ok=True)
    wid = sys.argv[2] if len(sys.argv) > 2 else "w"
    while True:
        took = None
        for t, c, e in all_jobs():
            cl = os.path.join(CLAIM, t + ".claim")
            if not missing(t) or not ready(c):
                continue
            if os.path.exists(cl) and time.time() - os.path.getmtime(cl) < 2400:
                continue
            open(cl, "w").write(wid)
            took = (t, c)
            break
        if not took:
            break
        subprocess.run(["bash", "-c", took[1]], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    print("WORKER DONE", wid)

if __name__ == "__main__":
    main()
