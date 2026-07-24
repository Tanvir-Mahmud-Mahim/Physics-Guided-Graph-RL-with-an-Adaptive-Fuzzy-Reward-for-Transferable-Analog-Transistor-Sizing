#!/usr/bin/env python3
"""Aggregate all run JSONs into tables (LaTeX fragments + numbers JSON)."""
import json, glob, os, sys
import numpy as np

RES = os.path.join(os.path.dirname(__file__), "..", "results")
OUT = os.path.join(RES, "agg")
os.makedirs(OUT, exist_ok=True)

def load(pattern):
    out = []
    for f in glob.glob(os.path.join(RES, pattern)):
        try:
            out.append(json.load(open(f)))
        except Exception:
            pass
    return out

def mstd(vals):
    v = np.array(vals, float)
    return float(v.mean()), float(v.std())

def main():
    runs = load("*.json")
    runs = [r for r in runs if isinstance(r, dict) and "method" in r]
    agg = {}

    # ---- main comparison ----
    main_methods = ["bo", "mace", "a2c", "ppo", "a2c_tskf", "ppo_tskf",
                    "gcnddpg", "gcnsac", "gcnsac_tskf", "gcnsac_tskf_pia",
                    "gcnsac_tskf_pia_ekv"]
    tbl = {}
    for ct in ("CT1", "CT2", "CT3"):
        for m in main_methods:
            sel = [r for r in runs if r["method"] == m and r["ct"] == ct
                   and r["pdk"] == "gf180" and not r.get("loaded")
                   and r["budget"] == 600]
            if sel:
                mu, sd = mstd([r["best_fom"] for r in sel])
                tbl[f"{ct}.{m}"] = {"mean": mu, "std": sd, "n": len(sel),
                                    "sims": [r["sims_used"] for r in sel],
                                    "wall": [r["wall_s"] for r in sel]}
    agg["main"] = tbl

    # best-design metrics (best seed per method)
    bm = {}
    for ct in ("CT1", "CT2", "CT3"):
        for m in main_methods:
            sel = [r for r in runs if r["method"] == m and r["ct"] == ct
                   and r["pdk"] == "gf180" and not r.get("loaded")
                   and r["budget"] == 600]
            if sel:
                b = max(sel, key=lambda r: r["best_fom"])
                bm[f"{ct}.{m}"] = {"fom": b["best_fom"], "met": b["best_met"],
                                   "params": b.get("best_params", {})}
    agg["bestmet"] = bm

    # ---- ablations ----
    ab = {}
    for m in ("gcnsac_tskf_pia", "gcnsac_tskf_pia_noper", "gcnsac_tskf_pia_noaug",
              "gcnsac_tskf_pia_nosimclr", "gcnsac_tskf_pia_t1", "gcnsac_tskf",
              "gcnsac", "gcnsac_tskf_pia_ekv"):
        sel = [r for r in runs if r["method"] == m and r["ct"] == "CT2"
               and r["pdk"] == "gf180" and not r.get("loaded")
               and r["budget"] == 600]
        if sel:
            mu, sd = mstd([r["best_fom"] for r in sel])
            ab[m] = {"mean": mu, "std": sd, "n": len(sel)}
    agg["ablation"] = ab

    # ---- technology transfer ----
    tr = {}
    for ct in ("CT1", "CT2", "CT3"):
        for pdk in ("sky130", "ptm65", "ptm45"):
            for cond in ("ft", "sc"):
                pat2 = glob.glob(os.path.join(RES, f"tr2_{ct}_{pdk}_{cond}_s*.json"))
                pat3 = glob.glob(os.path.join(RES, f"tr3_{ct}_{pdk}_ft_s*.json"))
                pat1 = glob.glob(os.path.join(RES, f"tr_{ct}_{pdk}_{cond}_s*.json"))
                use = pat3 if (cond == "ft" and len(pat3) >= 3) else pat1
                sel = [json.load(open(f)) for f in use]
                if sel:
                    mu, sd = mstd([r["best_fom"] for r in sel])
                    tr[f"{ct}.{pdk}.{cond}"] = {
                        "mean": mu, "std": sd, "n": len(sel),
                        "hist": [r["history"] for r in sel]}
    agg["transfer"] = tr

    # ---- topology transfer ----
    tt = {}
    for tag in ("CT2toCT3", "CT3toCT2"):
        dst = tag.split("to")[1]
        for cond, pat in (("ft", f"tt_{tag}_ft_s*.json"),
                          ("sc", f"tt_{dst}_sc_s*.json")):
            files = glob.glob(os.path.join(RES, pat))
            if cond == "ft":
                f3 = glob.glob(os.path.join(RES, f"tt3_{tag}_ft_s*.json"))
                if len(f3) >= 3:
                    files = f3

            sel = [json.load(open(f)) for f in files]
            if sel:
                mu, sd = mstd([r["best_fom"] for r in sel])
                tt[f"{tag}.{cond}"] = {"mean": mu, "std": sd, "n": len(sel),
                                       "hist": [r["history"] for r in sel]}
    agg["topo"] = tt

    # ---- cost accounting ----
    cost = {}
    for m in main_methods:
        sel = [r for r in runs if r["method"] == m and r["pdk"] == "gf180"
               and not r.get("loaded") and r["budget"] == 600]
        if sel:
            cost[m] = {"sims_mean": float(np.mean([r["sims_used"] for r in sel])),
                       "wall_mean": float(np.mean([r["wall_s"] for r in sel]))}
    agg["cost"] = cost

    json.dump(agg, open(os.path.join(OUT, "aggregate.json"), "w"), indent=1)
    # quick console summary
    for ct in ("CT1", "CT2", "CT3"):
        print(f"== {ct}")
        for m in main_methods:
            k = f"{ct}.{m}"
            if k in tbl:
                print(f"  {m:20s} {tbl[k]['mean']:.3f} +- {tbl[k]['std']:.3f}  (n={tbl[k]['n']})")

if __name__ == "__main__":
    main()
