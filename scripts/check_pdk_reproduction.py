#!/usr/bin/env python3
"""Check that your ngspice + PDK set-up reproduces the archived run logs.

Re-simulates the stored best design of a few archived runs for each
technology and compares the measured metrics and the FoM with the values in
the run log. Needs the unpacked run-log archive (`tar xzf results_logs.tar.gz`),
ngspice, and the model files under GCNSAC_PDK_ROOT (README, Section 3).

Usage (from the repository folder):
  python3 scripts/check_pdk_reproduction.py [--pdk ptm65 ptm45 sky130 gf180]
  python3 scripts/check_pdk_reproduction.py --pdk gf180 --gf180-pid

What to expect:
  ptm65, ptm45, sky130: every metric and the FoM equal the archived values
      exactly (the model files contain no random parameters).
  gf180: the re-simulated values differ from the archive, and from one call
      to the next. The GF180MCU file design.ngspice switches on transistor
      mismatch (sw_stat_mismatch = 1), so every ngspice run draws a random
      threshold-voltage offset for every transistor. ngspice 42 in batch mode
      seeds these draws with its own process ID (PID), which the run logs do
      not record. See README, Section 10.
  --gf180-pid: Linux only, needs `unshare` (util-linux) and either root or
      unprivileged user namespaces. Starts ngspice with the PIDs that the
      archived GF180MCU calibration file was made with (found for this
      repository; see README) and checks that results/cal/gf180.json is
      reproduced exactly. This confirms that your GF180MCU model files and
      ngspice build are the ones the archive was made with.
Exit status: 0 if every check that is expected to match does match.
"""
import argparse, json, os, shutil, subprocess, sys, tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
import numpy as np
from gcnsac import env as E
from gcnsac.pdk import PDK_ROOT

RES = os.path.join(ROOT, "results")
_SUBPROCESS_RUN = subprocess.run  # env.py uses the same subprocess module

# archived runs whose best design is re-simulated (tag = results/<tag>.json)
CASES = {
    "ptm65": ["tr_CT1_ptm65_sc_s0", "tr_CT3_ptm65_ft_s0"],
    "ptm45": ["tr_CT2_ptm45_sc_s0", "tr_CT3_ptm45_ft_s1"],
    "sky130": ["tr_CT1_sky130_ft_s2", "tr_CT2_sky130_sc_s1", "tr_CT3_sky130_sc_s0"],
    "gf180": ["bo_CT1_gf180_s0", "gcnsac_tskf_pia_CT2_gf180_s0", "a2c_CT3_gf180_s1"],
}
# ngspice process IDs of the 8 calibration sweeps (n: 1 gate sweep + 3 drain
# sweeps, then the same for p) that reproduce results/cal/gf180.json exactly.
GF180_CAL_PIDS = [10243, 10245, 10247, 10250, 10255, 10257, 10259, 10261]


def _run_ngspice_no_timeout(cmd, *a, **k):
    """env.simulate stops ngspice after 10 s; on a busy computer a SKY130 run
    can take longer, which would look like a failed design. Allow 120 s."""
    k["timeout"] = 120
    return _SUBPROCESS_RUN(cmd, *a, **k)


def resimulate(tag):
    """Re-simulate the best design stored in results/<tag>.json."""
    d = json.load(open(os.path.join(RES, f"{tag}.json")))
    env = E.CircuitEnv(d["ct"], d["pdk"])
    env.load_norm(os.path.join(RES, "norm", f"{d['ct']}_{d['pdk']}.json"))
    params = d["best_params"]
    env.topo.decode = lambda a: params  # use the stored sizes as they are
    orig = E.subprocess.run
    E.subprocess.run = _run_ngspice_no_timeout
    try:
        met, _ = env.simulate(np.zeros(env.topo.dim))
    finally:
        E.subprocess.run = orig
    return d, met, env.fom(met)


def compare(tag):
    d, met, fom = resimulate(tag)
    worst = 0.0
    same = fom == d["best_fom"]
    rows = []
    for k, a in d["best_met"].items():
        b = met.get(k, float("nan"))
        if a is None:
            ok = not np.isfinite(b)
            rel = 0.0 if ok else float("inf")
        else:
            ok = a == b
            rel = abs(b - a) / abs(a) if (np.isfinite(b) and a != 0) else (0.0 if ok else float("inf"))
        same &= ok
        worst = max(worst, rel)
        rows.append((k, a, b))
    return same, d["best_fom"], fom, worst, rows


# ------------------------- GF180MCU PID check -------------------------

def _pid_worker(outfile):
    """Runs inside a new PID namespace: redo the GF180MCU calibration with
    ngspice started at the PIDs in GF180_CAL_PIDS."""
    from gcnsac import surrogate as S
    from gcnsac.pdk import get_pdk
    targets = iter(GF180_CAL_PIDS)
    orig = S.subprocess.run
    state = {"last": 0}  # highest PID known to be used in this namespace

    def run(cmd, *a, **k):
        t = next(targets)
        # Start short-lived /bin/true processes until PID t-1 is used, so that
        # ngspice (started directly, as in surrogate.py) gets PID t. Nothing
        # else runs in this new PID namespace.
        if t - 1 - state["last"] > 50:  # large gap: a bash loop is faster
            r = orig(["bash", "-c", f'n=$$; while [ "$n" -lt {t - 1} ]; do : & n=$!; '
                      'wait "$n"; done; echo "$n"'], capture_output=True, text=True)
            state["last"] = int(r.stdout.split()[-1])
        while state["last"] < t - 1:
            pid = os.posix_spawn("/bin/true", ["true"], os.environ)
            os.waitpid(pid, 0)
            state["last"] = pid
        if state["last"] != t - 1:
            raise RuntimeError(f"cannot give ngspice PID {t} (last used PID {state['last']})")
        k["timeout"] = 600
        r = orig(cmd, *a, **k)
        # ngspice 42 starts one extra thread, which uses the next PID number
        state["last"] = t + 1
        return r

    S.subprocess.run = run
    tmp = tempfile.mkdtemp()
    S.CAL_DIR = tmp
    try:
        out = S.calibrate(get_pdk("gf180"), force=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    json.dump(out, open(outfile, "w"))


def gf180_pid_check():
    arch = json.load(open(os.path.join(RES, "cal", "gf180.json")))
    fd, outfile = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    me = [sys.executable, os.path.abspath(__file__), "--_pid-worker", outfile]
    tries = [["unshare", "--pid", "--fork", "--mount-proc"],
             ["unshare", "--user", "--map-root-user", "--pid", "--fork", "--mount-proc"]]
    got, err = None, "unshare not found"
    for pre in tries:
        if shutil.which(pre[0]) is None:
            break
        r = subprocess.run(pre + me, capture_output=True, text=True)
        if r.returncode == 0:
            got = json.load(open(outfile))
            break
        err = (r.stderr.strip().splitlines() or [""])[-1]
    os.unlink(outfile)
    if got is None:
        print("  gf180 calibration with fixed PIDs: could not run "
              f"(needs Linux `unshare`; last error: {err})")
        return None
    ok = got == arch
    print(f"  gf180 calibration (ngspice PIDs {GF180_CAL_PIDS}) "
          f"{'MATCH' if ok else 'MISMATCH'} with results/cal/gf180.json")
    for t in ("n", "p"):
        for k in ("vth", "kp", "n_ekv"):
            print(f"    {t} {k:5s} archived {arch[t][k]!r:24s} now {got[t][k]!r}")
        print(f"    {t} lam   archived {arch[t]['lam']} now {got[t]['lam']}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pdk", nargs="+", default=list(CASES), choices=list(CASES))
    ap.add_argument("--gf180-pid", action="store_true",
                    help="also redo the GF180MCU calibration with the archived ngspice PIDs")
    ap.add_argument("--_pid-worker", dest="pid_worker", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.pid_worker:
        _pid_worker(args.pid_worker)
        return 0
    if not os.path.isdir(os.path.join(RES, "norm")):
        sys.exit("results/ not found: run `tar xzf results_logs.tar.gz` in the repository folder first")
    r = subprocess.run(["ngspice", "-v"], capture_output=True, text=True)
    ver = next((l.strip("* ").strip() for l in r.stdout.splitlines() if "ngspice-" in l), "?")
    print(f"ngspice: {ver}")
    print(f"GCNSAC_PDK_ROOT: {PDK_ROOT}")
    all_ok = True
    for pdk in args.pdk:
        print(f"\n[{pdk}]")
        for tag in CASES[pdk]:
            same, fa, fb, worst, rows = compare(tag)
            if pdk == "gf180":
                status = "MATCH" if same else "MISMATCH (expected for gf180: random mismatch, see --help)"
            else:
                status = "MATCH" if same else "MISMATCH"
                all_ok &= same
            print(f"  {tag}: {status}; FoM archived {fa:.6f}, now {fb:.6f}; "
                  f"largest relative metric difference {worst:.3g}")
            if not same:
                for k, a, b in rows:
                    print(f"    {k:7s} archived {a!r:>24}  now {b!r}")
        if pdk == "gf180" and args.gf180_pid:
            ok = gf180_pid_check()
            all_ok &= bool(ok)
    print("\nRESULT:", "all expected matches found" if all_ok else "some expected matches FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
