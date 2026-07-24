#!/usr/bin/env python3
"""Extended campaign: 5 seeds for all methods, 1500-step pretrains for the
proposed method, and transfer redone from the deeper pretraining.
Same chunked, resume-aware execution model as chunk.py."""
import os, subprocess, sys

RES = "results"

def missing(tag):
    return not os.path.exists(os.path.join(RES, tag + ".json"))

def est(method, budget, ct="CT2"):
    k = 1.5 if ct == "CT3" else 1.0
    if method in ("bo", "mace"):
        return (220 if budget <= 600 else 600) * k
    if method.startswith(("a2c", "ppo")):
        return (140 if budget == 600 else 40) * k
    base = {"gcnddpg": 220, "gcnsac": 280, "gcnsac_tskf": 310}.get(method, 340)
    if budget == 600:
        return base * k
    if budget == 1500:
        return base * 2.5 * k
    return (base * 0.3 + 30) * k

def all_jobs():
    jobs = []
    # 1) seeds 3, 4 for all methods on all circuits (600 evals)
    for ct in ("CT1", "CT2", "CT3"):
        for m in ("bo mace a2c ppo a2c_tskf ppo_tskf gcnddpg gcnsac "
                  "gcnsac_tskf gcnsac_tskf_pia").split():
            for s in (3, 4):
                tag = f"{m}_{ct}_gf180_s{s}"
                jobs.append((tag, f"python3 scripts/run.py --method {m} "
                             f"--ct {ct} --seed {s} --budget 600 --pdk gf180",
                             est(m, 600, ct)))
    # 2) ablations to 5 seeds
    for m in ("gcnsac_tskf_pia_noper gcnsac_tskf_pia_noaug "
              "gcnsac_tskf_pia_nosimclr gcnsac_tskf_pia_t1").split():
        for s in (3, 4):
            tag = f"{m}_CT2_gf180_s{s}"
            jobs.append((tag, f"python3 scripts/run.py --method {m} --ct CT2 "
                         f"--seed {s} --budget 600 --pdk gf180",
                         est("gcnsac_tskf_pia", 600, "CT2")))
    # 2b) EKV-surrogate configuration on CT-2 (5 seeds)
    for s in (0, 1, 2, 3, 4):
        tag = f"gcnsac_tskf_pia_ekv_CT2_gf180_s{s}"
        jobs.append((tag, f"python3 scripts/run.py --method gcnsac_tskf_pia_ekv "
                     f"--ct CT2 --seed {s} --budget 600 --pdk gf180",
                     est("gcnsac_tskf_pia", 600, "CT2")))
    # 3) deeper pretraining for the proposed method (1500 evals)
    for ct in ("CT1", "CT2", "CT3"):
        for s in (0, 1, 2):
            tag = f"pia1500_{ct}_s{s}"
            jobs.append((tag, f"python3 scripts/run.py --method gcnsac_tskf_pia "
                         f"--ct {ct} --seed {s} --budget 1500 --pdk gf180 "
                         f"--tag {tag}", est("gcnsac_tskf_pia", 1500, ct)))
    # 4) technology transfer from the 1500-step pretraining (encoder reuse)
    for ct in ("CT1", "CT2", "CT3"):
        for pdk in ("sky130", "ptm65", "ptm45"):
            for s in (0, 1, 2):
                tag = f"tr2_{ct}_{pdk}_ft_s{s}"
                jobs.append((tag, f"python3 scripts/run.py --method "
                             f"gcnsac_tskf_pia --ct {ct} --pdk {pdk} "
                             f"--seed {s} --budget 150 "
                             f"--load results/w/pia1500_{ct}_s{s}.pt "
                             f"--enc_only --tag {tag}",
                             est("gcnsac_tskf_pia", 150, ct)))
    # 5) topology transfer from the 1500-step pretraining
    for src, dst in (("CT2", "CT3"), ("CT3", "CT2")):
        for s in (0, 1, 2):
            tag = f"tt2_{src}to{dst}_ft_s{s}"
            jobs.append((tag, f"python3 scripts/run.py --method "
                         f"gcnsac_tskf_pia --ct {dst} --pdk gf180 --seed {s} "
                         f"--budget 150 --load results/w/pia1500_{src}_s{s}.pt "
                         f"--enc_only --tag {tag}",
                         est("gcnsac_tskf_pia", 150, dst)))
    return jobs

def ready(tag, cmd):
    """Transfer jobs are ready only once their pretrained weights exist."""
    if "--load " in cmd:
        w = cmd.split("--load ")[1].split()[0]
        return os.path.exists(w)
    return True

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "background"
    jobs = [(t, c, e) for (t, c, e) in all_jobs()
            if missing(t) and ready(t, c)]
    if not jobs:
        print("EXT_ALL_COMPLETE")
        return
    if mode == "count":
        print(len(jobs), "missing")
        return
    if mode == "background":
        # two persistent worker lanes over the whole remaining list
        lanes = [[], []]
        lt = [0.0, 0.0]
        for t, c, e in jobs:
            i = 0 if lt[0] <= lt[1] else 1
            lanes[i].append(c); lt[i] += e
        procs = []
        for lane in lanes:
            script = " ; ".join(lane)
            procs.append(subprocess.Popen(["bash", "-c", script],
                                          stdout=subprocess.DEVNULL,
                                          stderr=subprocess.DEVNULL))
        for p in procs:
            p.wait()
        print("EXT_ALL_COMPLETE")
    else:  # chunk mode with time budget
        budget_s = float(mode)
        lanes = [[], []]
        lt = [0.0, 0.0]
        picked = []
        for t, c, e in jobs:
            i = 0 if lt[0] <= lt[1] else 1
            if lt[i] + e > budget_s:
                continue
            lanes[i].append(c); lt[i] += e; picked.append(t)
            if min(lt) >= budget_s * 0.9:
                break
        if not picked:
            lanes[0].append(jobs[0][1]); picked.append(jobs[0][0])
        procs = [subprocess.Popen(["bash", "-c", " && ".join(l)],
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)
                 for l in lanes if l]
        for p in procs:
            p.wait()
        done = sum(0 if missing(t) else 1 for t in picked)
        rem = sum(1 for (t, c, e) in all_jobs() if missing(t))
        print(f"chunk: {done}/{len(picked)}; {rem} remain")

if __name__ == "__main__":
    main()
