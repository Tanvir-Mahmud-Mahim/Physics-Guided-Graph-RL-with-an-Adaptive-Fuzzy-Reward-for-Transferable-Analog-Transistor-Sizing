#!/usr/bin/env python3
"""Looping worker: repeatedly claims the next ready+unclaimed job until none
remain. Claim files prevent two workers from taking the same job."""
import os, subprocess, sys, time
sys.path.insert(0, "scripts")
from chunk2 import all_jobs, missing, ready

CLAIM = "results/claims"
os.makedirs(CLAIM, exist_ok=True)
wid = sys.argv[1] if len(sys.argv) > 1 else "w0"
while True:
    took = None
    for t, c, e in all_jobs():
        cl = os.path.join(CLAIM, t + ".claim")
        if not missing(t) or not ready(t, c):
            continue
        if os.path.exists(cl) and time.time() - os.path.getmtime(cl) < 3600:
            continue
        open(cl, "w").write(wid)
        took = (t, c)
        break
    if not took:
        break
    subprocess.run(["bash", "-c", took[1]], stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
print("WORKER DONE", wid)
