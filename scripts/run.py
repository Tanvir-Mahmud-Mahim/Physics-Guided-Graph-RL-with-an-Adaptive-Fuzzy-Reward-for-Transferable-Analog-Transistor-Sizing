#!/usr/bin/env python3
"""Single experiment runner.

Usage:
  python3 scripts/run.py --method gcnsac_tskf_pia --ct CT2 --pdk gf180 \
      --seed 0 --budget 600 [--load results/w/<tag>.pt] [--tag mytag]

Methods:
  bo, mace, a2c, ppo, a2c_tskf, ppo_tskf, gcnddpg,
  gcnsac, gcnsac_tskf, gcnsac_tskf_pia,
  ablations: gcnsac_tskf_pia_noper / _noaug / _nosimclr / _t1 / (pia off = gcnsac_tskf)
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gcnsac.env import CircuitEnv, SimCounter
from gcnsac.tskf import IT2TSKF
from gcnsac.surrogate import calibrate, PhysicsSurrogate
from gcnsac.agents import GCNSAC, GCNDDPG, NGAgent, build_graph
from gcnsac.baselines import run_bo

RES = os.path.join(os.path.dirname(__file__), "..", "results")
NORM_SAMPLES = 300
EP_LEN = 4

def get_norm(env, ct, pdk):
    os.makedirs(os.path.join(RES, "norm"), exist_ok=True)
    path = os.path.join(RES, "norm", f"{ct}_{pdk}.json")
    if os.path.exists(path):
        env.load_norm(path)
        return 0
    env.calibrate(n_samples=NORM_SAMPLES, seed=1234, save=path)
    return NORM_SAMPLES

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--ct", required=True)
    ap.add_argument("--pdk", default="gf180")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=600)
    ap.add_argument("--load", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--enc_lr_scale", type=float, default=0.1)
    ap.add_argument("--enc_only", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    counter = SimCounter()
    env = CircuitEnv(args.ct, args.pdk, counter=counter)
    norm_sims = get_norm(env, args.ct, args.pdk)
    method = args.method
    rng = np.random.default_rng(args.seed)
    tag = args.tag or f"{method}_{args.ct}_{args.pdk}_s{args.seed}"
    os.makedirs(RES, exist_ok=True)
    os.makedirs(os.path.join(RES, "w"), exist_ok=True)

    hist, best, best_met, best_p = [], -1e9, None, None

    def record(step, f, met=None, p=None):
        nonlocal best, best_met, best_p
        if f > best:
            best, best_met, best_p = f, met, p
        hist.append((step, best))

    signs = [1 if env.topo.directions[m] > 0 else -1 for m in env.topo.metrics]

    if method in ("bo", "mace"):
        orig_step = env.step
        state = {"n": 0}
        def wrapped(a):
            met, p, f = orig_step(a)
            state["n"] += 1
            record(state["n"], f, met, p)
            return met, p, f
        env.step = wrapped
        run_bo(env, args.budget, seed=args.seed, mace=(method == "mace"))
    elif method.startswith("a2c") or method.startswith("ppo"):
        kind = "ppo" if method.startswith("ppo") else "a2c"
        use_tskf = method.endswith("_tskf")
        adim = env.topo.dim
        agent = NGAgent(adim, adim, kind=kind, seed=args.seed)
        tskf = IT2TSKF(len(env.topo.metrics), seed=args.seed) if use_tskf else None
        s = np.zeros(adim, dtype=np.float32)
        step = 0
        while step < args.budget:
            for t in range(EP_LEN):
                if step >= args.budget:
                    break
                a, logp, v = agent.act(s)
                met, p, f = env.step(a)
                nz = env.normalized(met)
                if use_tskf:
                    r = tskf.forward([nz[m] for m in env.topo.metrics], signs)
                    tskf.adapt(f / max(len(env.topo.metrics), 1))
                else:
                    r = f
                agent.store(s, a, r, logp, v)
                s = a.astype(np.float32)
                step += 1
                record(step, f, met, p)
            agent.update()
    else:
        # graph-based agents
        cal = calibrate(env.pdk, n_sims_counter=counter)
        adim = env.topo.dim
        use_tskf = "tskf" in method
        use_pia = "pia" in method
        use_per = "noper" not in method
        use_aug = "noaug" not in method
        use_simclr = "nosimclr" not in method
        type1 = "_t1" in method
        surrogate = None
        if method == "gcnddpg":
            agent = GCNDDPG(env.topo, cal, adim, seed=args.seed)
        else:
            if use_pia:
                surrogate = PhysicsSurrogate(
                    env, cal, model="ekv" if "ekv" in method else "sq")
            agent = GCNSAC(env.topo, cal, adim, seed=args.seed, use_per=use_per,
                           use_aug=use_aug, use_simclr=use_simclr,
                           surrogate=surrogate)
        if args.load:
            sd = torch.load(args.load, weights_only=True)
            agent.enc.load_state_dict(sd["enc"])
            if not args.enc_only:
                try:
                    agent.actor.load_state_dict(sd["actor"])
                    if "q1" in sd and hasattr(agent, "q1"):
                        agent.q1.load_state_dict(sd["q1"]); agent.q2.load_state_dict(sd["q2"])
                        agent.q1t.load_state_dict(sd["q1"]); agent.q2t.load_state_dict(sd["q2"])
                except RuntimeError:
                    pass  # action dims differ: encoder-only reuse
            # reduced encoder learning rate during fine-tuning
            if hasattr(agent, "opt_a"):
                agent.opt_a = torch.optim.Adam([
                    {"params": agent.actor.parameters(), "lr": 3e-4},
                    {"params": agent.enc.parameters(),
                     "lr": 3e-4 * args.enc_lr_scale}])
        tskf = IT2TSKF(len(env.topo.metrics), seed=args.seed,
                       type1=type1) if use_tskf else None
        p = env.topo.decode(np.zeros(adim))
        step = 0
        while step < args.budget:
            s_graph = build_graph(env.topo, p, cal)
            for t in range(EP_LEN):
                if step >= args.budget:
                    break
                a, s_graph = agent.act(p)
                met, p2, f = env.step(a)
                nz = env.normalized(met)
                if use_tskf:
                    r = tskf.forward([nz[m] for m in env.topo.metrics], signs)
                    tskf.adapt(f / max(len(env.topo.metrics), 1))
                else:
                    r = f
                s2_graph = build_graph(env.topo, p2, cal)
                agent.store(s_graph, a, r, s2_graph, float(t == EP_LEN - 1))
                p = p2
                step += 1
                record(step, f, met, p2)
                agent.update(1)
            p = env.topo.decode(rng.uniform(-1, 1, adim) * 0.2)  # soft restart
        sd = {"enc": agent.enc.state_dict(), "actor": agent.actor.state_dict()}
        if hasattr(agent, "q1"):
            sd["q1"] = agent.q1.state_dict(); sd["q2"] = agent.q2.state_dict()
        torch.save(sd, os.path.join(RES, "w", f"{tag}.pt"))
        if use_tskf:
            np.save(os.path.join(RES, "w", f"{tag}_tskfhist.npy"),
                    np.array(tskf.hist))

    out = {"method": method, "ct": args.ct, "pdk": args.pdk, "seed": args.seed,
           "budget": args.budget, "norm_sims": norm_sims,
           "sims_used": counter.n, "wall_s": time.time() - t0,
           "best_fom": best if best > -1e8 else (hist[-1][1] if hist else 0),
           "best_met": {k: (None if (v is None or not np.isfinite(v)) else float(v))
                        for k, v in (best_met or {}).items() if k != "_dead"},
           "best_params": {k: float(v) for k, v in (best_p or {}).items()},
           "history": [(int(s), float(b)) for s, b in hist],
           "loaded": bool(args.load)}
    with open(os.path.join(RES, f"{tag}.json"), "w") as fh:
        json.dump(out, fh)
    print(json.dumps({k: out[k] for k in
                      ("method", "ct", "pdk", "seed", "best_fom",
                       "sims_used", "wall_s")}))

if __name__ == "__main__":
    main()
