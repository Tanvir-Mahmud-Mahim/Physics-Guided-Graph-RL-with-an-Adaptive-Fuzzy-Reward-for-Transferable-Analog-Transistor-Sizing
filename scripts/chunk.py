#!/usr/bin/env python3
"""Emit and run the next batch of missing experiment jobs, sized to finish
inside one ~9-minute foreground window with 2 parallel workers."""
import os, subprocess, sys, json

RES = "results"

def missing(tag):
    return not os.path.exists(os.path.join(RES, tag + ".json"))

def est(method, budget, ct="CT2"):
    """Rough wall estimate (s) per run, 1 core."""
    k = 1.5 if ct == "CT3" else 1.0
    if method in ("bo", "mace"):
        return 220 * k
    if method.startswith(("a2c", "ppo")):
        return (140 if budget == 600 else 40) * k
    base = {"gcnddpg": 220, "gcnsac": 280, "gcnsac_tskf": 310}.get(method, 340)
    return base * k if budget == 600 else (base * 0.3 + 30) * k

def all_jobs():
    jobs = []  # (tag, cmd, est)
    for ct in ("CT1", "CT2", "CT3"):
        for m in ("bo mace a2c ppo a2c_tskf ppo_tskf gcnddpg gcnsac "
                  "gcnsac_tskf gcnsac_tskf_pia").split():
            for s in (0, 1, 2):
                tag = f"{m}_{ct}_gf180_s{s}"
                cmd = (f"python3 scripts/run.py --method {m} --ct {ct} "
                       f"--seed {s} --budget 600 --pdk gf180")
                jobs.append((tag, cmd, est(m, 600, ct)))
    for m in ("gcnsac_tskf_pia_noper gcnsac_tskf_pia_noaug "
              "gcnsac_tskf_pia_nosimclr gcnsac_tskf_pia_t1").split():
        for s in (0, 1, 2):
            tag = f"{m}_CT2_gf180_s{s}"
            cmd = (f"python3 scripts/run.py --method {m} --ct CT2 "
                   f"--seed {s} --budget 600 --pdk gf180")
            jobs.append((tag, cmd, est("gcnsac_tskf_pia", 600, "CT2")))
    for ct in ("CT1", "CT2", "CT3"):
        for pdk in ("sky130", "ptm65", "ptm45"):
            for s in (0, 1, 2):
                t1 = f"tr_{ct}_{pdk}_ft_s{s}"
                jobs.append((t1, f"python3 scripts/run.py --method gcnsac_tskf_pia "
                             f"--ct {ct} --pdk {pdk} --seed {s} --budget 150 "
                             f"--load results/w/gcnsac_tskf_pia_{ct}_gf180_s{s}.pt "
                             f"--tag {t1}", est("gcnsac_tskf_pia", 150, ct)))
                t2 = f"tr_{ct}_{pdk}_sc_s{s}"
                jobs.append((t2, f"python3 scripts/run.py --method gcnsac_tskf_pia "
                             f"--ct {ct} --pdk {pdk} --seed {s} --budget 150 "
                             f"--tag {t2}", est("gcnsac_tskf_pia", 150)))
    for src, dst in (("CT2", "CT3"), ("CT3", "CT2")):
        for s in (0, 1, 2):
            t1 = f"tt_{src}to{dst}_ft_s{s}"
            jobs.append((t1, f"python3 scripts/run.py --method gcnsac_tskf_pia "
                         f"--ct {dst} --pdk gf180 --seed {s} --budget 150 "
                         f"--load results/w/gcnsac_tskf_pia_{src}_gf180_s{s}.pt "
                         f"--tag {t1}", est("gcnsac_tskf_pia", 150)))
            t2 = f"tt_{dst}_sc_s{s}"
            jobs.append((t2, f"python3 scripts/run.py --method gcnsac_tskf_pia "
                         f"--ct {dst} --pdk gf180 --seed {s} --budget 150 "
                         f"--tag {t2}", est("gcnsac_tskf_pia", 150)))
    return jobs

def gen_norm_if_needed():
    """Sequentially generate any missing normalization stats (fast)."""
    import importlib
    sys.path.insert(0, ".")
    from gcnsac.env import CircuitEnv
    made = []
    for ct in ("CT1", "CT2", "CT3"):
        for pdk in ("sky130", "ptm65", "ptm45"):
            p = f"results/norm/{ct}_{pdk}.json"
            if not os.path.exists(p):
                CircuitEnv(ct, pdk).calibrate(n_samples=300, seed=1234, save=p)
                made.append(f"{ct}_{pdk}")
    return made

def main():
    budget_s = float(sys.argv[1]) if len(sys.argv) > 1 else 480.0
    jobs = [(t, c, e) for (t, c, e) in all_jobs() if missing(t)]
    if not jobs:
        print("ALL_RUNS_COMPLETE")
        return
    # transfer runs need norm stats first
    if any(t.startswith(("tr_", "tt_")) for t, _, _ in jobs[:8]):
        made = gen_norm_if_needed()
        if made:
            print("norm generated:", ",".join(made))
    # greedily fill two worker lanes up to the time budget
    lanes = [[], []]
    lane_t = [0.0, 0.0]
    picked = []
    for t, c, e in jobs:
        i = 0 if lane_t[0] <= lane_t[1] else 1
        if lane_t[i] + e > budget_s:
            continue
        lanes[i].append((t, c)); lane_t[i] += e; picked.append(t)
        if min(lane_t) >= budget_s * 0.9:
            break
    if not picked:  # single long job fallback
        t, c, e = jobs[0]
        lanes[0].append((t, c)); picked.append(t)
    procs = []
    for lane in lanes:
        if not lane:
            continue
        script = " && ".join(c for _, c in lane)
        procs.append(subprocess.Popen(["bash", "-c", script],
                                      stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL))
    for pr in procs:
        pr.wait()
    done = sum(0 if missing(t) else 1 for t in picked)
    total_missing = sum(1 for (t, c, e) in all_jobs() if missing(t))
    print(f"chunk: {done}/{len(picked)} finished; {total_missing} runs remain")

if __name__ == "__main__":
    main()
