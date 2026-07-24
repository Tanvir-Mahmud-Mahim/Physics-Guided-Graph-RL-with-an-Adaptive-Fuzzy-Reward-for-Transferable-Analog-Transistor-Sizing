#!/usr/bin/env python3
"""Generate paper figures (PDF) from real run logs.

F5  : learning curves, best-so-far unified FoM vs simulator evaluations
F6  : IT2-TSKF adaptation dynamics (error, consequent mass, firing level)
F8a : technology transfer (fine-tune vs from-scratch)
F8b : topology transfer
F9  : ablation bars
"""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(RES, "figs")
os.makedirs(FIG, exist_ok=True)

# dataviz palette (light mode, fixed categorical order)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
     "#4a3aa7", "#e34948"]
INK, MUT, GRID = "#0b0b0b", "#898781", "#e1e0d9"

plt.rcParams.update({
    "font.size": 8, "font.family": "DejaVu Sans",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
    "xtick.color": MUT, "ytick.color": MUT, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 150})

def band(ax, runs, color, label, budget=600):
    xs = np.arange(1, budget + 1)
    ys = []
    for h in runs:
        arr = np.full(budget, np.nan)
        for s, b in h:
            if 1 <= s <= budget:
                arr[s - 1] = b
        # forward fill
        last = np.nan
        for i in range(budget):
            if np.isnan(arr[i]):
                arr[i] = last
            last = arr[i]
        ys.append(arr)
    Y = np.array(ys, float)
    mu = np.nanmean(Y, 0); sd = np.nanstd(Y, 0)
    ax.plot(xs, mu, color=color, lw=1.4, label=label)
    ax.fill_between(xs, mu - sd, mu + sd, color=color, alpha=0.15, lw=0)

def hist_of(method, ct, pdk="gf180", budget=600):
    out = []
    for f in glob.glob(os.path.join(RES, f"{method}_{ct}_{pdk}_s*.json")):
        r = json.load(open(f))
        if r["budget"] == budget and not r.get("loaded"):
            out.append(r["history"])
    return out

def fig_learning():
    methods = [("bo", "BO", C[3]), ("mace", "MACE", C[4]),
               ("gcnddpg", "GCN-RL (DDPG)", C[2]),
               ("gcnsac", "GCN-SAC", C[5]),
               ("gcnsac_tskf", "GCN-SAC-TSKF", C[1]),
               ("gcnsac_tskf_pia_ekv", "PIA-EKV (ours)", C[0])]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2), constrained_layout=True)
    for ax, ct, ttl in zip(axes, ("CT1", "CT2", "CT3"),
                           ("(a) CT-1 two-stage voltage amp",
                            "(b) CT-2 two-stage TIA",
                            "(c) CT-3 three-stage TIA")):
        for m, lbl, col in methods:
            runs = hist_of(m, ct)
            if runs:
                band(ax, runs, col, lbl)
        ax.set_xlabel("simulator evaluations")
        ax.set_title(ttl, fontsize=8)
    axes[0].set_ylabel("best unified FoM")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22),
               fontsize=7)
    fig.savefig(os.path.join(FIG, "F5_learning.pdf"), bbox_inches="tight")
    plt.close(fig)

def fig_tskf():
    fig, axes = plt.subplots(3, 1, figsize=(3.45, 4.4), constrained_layout=True)
    for ax, ct, ttl in zip(axes, ("CT1", "CT2", "CT3"),
                           ("(a) CT-1", "(b) CT-2", "(c) CT-3")):
        fs = sorted(glob.glob(os.path.join(
            RES, "w", f"gcnsac_tskf_pia_{ct}_gf180_s0_tskfhist.npy")))
        if not fs:
            continue
        H = np.load(fs[0])  # (T, 4): err, |W|mean, wbar, r
        t = np.arange(len(H))
        ax.plot(t, np.abs(H[:, 0]), color=C[1], lw=1.0, label="|shaping error| $|e|$")
        ax.plot(t, H[:, 1], color=C[0], lw=1.0, label=r"consequent mass $\overline{|w^{(k)}|}$")
        ax.plot(t, H[:, 2], color=C[2], lw=1.0, label=r"mean firing $\bar{w}$")
        ax.set_title(ttl, fontsize=8)
        ax.set_ylabel("magnitude")
    axes[2].set_xlabel("training step")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=1, loc="outside upper center", fontsize=6.6)
    fig.savefig(os.path.join(FIG, "F6_tskf.pdf"), bbox_inches="tight")
    plt.close(fig)

def _transfer_panel(ax, patt_ft, patt_sc, ttl, budget=150):
    ft = [json.load(open(f))["history"] for f in glob.glob(patt_ft)]
    sc = [json.load(open(f))["history"] for f in glob.glob(patt_sc)]
    if ft:
        band(ax, ft, C[0], "with transfer", budget)
    if sc:
        band(ax, sc, C[1], "no transfer", budget)
    ax.set_title(ttl, fontsize=7.5)
    ax.set_xlabel("simulator evaluations")

def fig_transfer():
    import glob as _g
    fig, axes = plt.subplots(4, 3, figsize=(7.0, 6.6), constrained_layout=True)
    for i, ct in enumerate(("CT1", "CT2", "CT3")):
        for j, (pdk, nm) in enumerate((("sky130", "SKY130 (130 nm)"),
                                       ("ptm65", "PTM 65 nm"),
                                       ("ptm45", "PTM 45 nm"))):
            ftpat = os.path.join(RES, f"tr3_{ct}_{pdk}_ft_s*.json")
            if len(_g.glob(ftpat)) < 3:
                ftpat = os.path.join(RES, f"tr_{ct}_{pdk}_ft_s*.json")
            _transfer_panel(axes[i, j], ftpat,
                            os.path.join(RES, f"tr_{ct}_{pdk}_sc_s*.json"),
                            f"({chr(97+3*i+j)}) {ct.replace('CT','CT-')} "
                            f"$\\rightarrow$ {nm}")
            if j == 0:
                axes[i, j].set_ylabel("best unified FoM")
            if i < 3:
                axes[i, j].set_xlabel("")
    # fourth row: topology transfer (encoder reuse)
    for j, (tag, ttl) in enumerate((("CT2toCT3", "(j) CT-2 $\\rightarrow$ CT-3 (topology)"),
                                    ("CT3toCT2", "(k) CT-3 $\\rightarrow$ CT-2 (topology)"))):
        dst = tag.split("to")[1]
        _transfer_panel(axes[3, j], os.path.join(RES, f"tt_{tag}_ft_s*.json"),
                        os.path.join(RES, f"tt_{dst}_sc_s*.json"), ttl)
    axes[3, 0].set_ylabel("best unified FoM")
    axes[3, 2].axis("off")
    h, l = axes[0, 0].get_legend_handles_labels()
    axes[3, 2].legend(h, l, ncol=1, loc="center", fontsize=9)
    fig.savefig(os.path.join(FIG, "F8a_tech_transfer.pdf"), bbox_inches="tight")
    plt.close(fig)

def fig_topo():
    import glob as _g
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.0), constrained_layout=True)
    import glob as _g
    for ax, tag, ttl in ((axes[0], "CT2toCT3", "CT-2 $\\rightarrow$ CT-3"),
                         (axes[1], "CT3toCT2", "CT-3 $\\rightarrow$ CT-2")):
        dst = tag.split("to")[1]
        ftpat = os.path.join(RES, f"tt3_{tag}_ft_s*.json")
        if len(_g.glob(ftpat)) < 3:
            ftpat = os.path.join(RES, f"tt_{tag}_ft_s*.json")
        _transfer_panel(ax, ftpat,
                        os.path.join(RES, f"tt_{dst}_sc_s*.json"), ttl)
    axes[0].set_ylabel("best unified FoM")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.15),
               fontsize=8)
    fig.savefig(os.path.join(FIG, "F8b_topo_transfer.pdf"), bbox_inches="tight")
    plt.close(fig)

def fig_ablation():
    agg = json.load(open(os.path.join(RES, "agg", "aggregate.json")))
    ab = agg["ablation"]
    order = [("gcnsac_tskf_pia_ekv", "Full (EKV surrogate, ours)", C[0], True),
             ("gcnsac_tskf_pia", "long-channel surrogate", C[0], False),
             ("gcnsac_tskf", "no adjoint guidance", C[1], True),
             ("gcnsac_tskf_pia_t1", "type-1 reward (no FOU)", C[1], True),
             ("gcnsac", "standard fixed reward", C[1], True),
             ("gcnsac_tskf_pia_noper", "no PER", C[2], False),
             ("gcnsac_tskf_pia_noaug", "no augmentation", C[2], False),
             ("gcnsac_tskf_pia_nosimclr", "no contrastive", C[2], False)]
    keys = [(k, l, c, hl) for k, l, c, hl in order if k in ab]
    full = ab["gcnsac_tskf_pia_ekv"]["mean"]
    y = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(3.5, 2.35), constrained_layout=True)
    ax.axvline(full, color=C[0], lw=1.0, ls=(0, (4, 3)), alpha=0.7)
    for yi, (k, l, c, hl) in zip(y, keys):
        m, s = ab[k]["mean"], ab[k]["std"]
        ax.errorbar([m], [yi], xerr=[s], fmt="o", color=c, ms=6 if hl else 5,
                    capsize=3, lw=1.3,
                    markeredgecolor="white", markeredgewidth=0.7)
        d = m - full
        txt = f"{m:.2f}" if k == "gcnsac_tskf_pia_ekv" else f"{d:+.2f}"
        ax.annotate(txt, (m + s, yi), textcoords="offset points",
                    xytext=(5, 0), ha="left", va="center", fontsize=6.6,
                    color=c, weight="bold" if hl else "normal")
    ax.set_yticks(y, [l for _, l, _, _ in keys], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(3.08, 3.72)
    ax.set_xticks([3.1, 3.2, 3.3, 3.4, 3.5, 3.6])
    ax.set_xlabel("best unified FoM (CT-2, 600 evals, 5 seeds)", fontsize=7.5)
    fig.savefig(os.path.join(FIG, "F9_ablation.pdf"), bbox_inches="tight")
    plt.close(fig)

def fig_convergence():
    """Novelty advantage: early-training convergence of the two new learning
    signals against the standard fixed reward (mean best-so-far FoM)."""
    sel = [("gcnsac", "standard fixed reward", C[5]),
           ("gcnsac_tskf", "+ IT2-TSKF reward", C[1]),
           ("gcnsac_tskf_pia_ekv", "+ adjoint physics (ours)", C[0])]
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.9), constrained_layout=True)
    for ax, ct, ttl, thr in ((axes[0], "CT1", "(a) CT-1", None),
                             (axes[1], "CT2", "(b) CT-2", None)):
        for m, lbl, col in sel:
            runs = hist_of(m, ct)
            if runs:
                band(ax, runs, col, lbl, budget=200)
        ax.set_title(ttl, fontsize=8)
        ax.set_ylabel("best unified FoM")
        ax.set_xlim(0, 200)
    axes[1].set_xlabel("simulator evaluations")
    axes[0].legend(fontsize=6.4, loc="lower right", handlelength=1.4)
    fig.savefig(os.path.join(FIG, "F6_convergence.pdf"), bbox_inches="tight")
    plt.close(fig)

def fig_tskf_supp():
    """Raw IT2-TSKF adaptation traces (supplementary)."""
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.0), constrained_layout=True)
    for ax, ct, ttl in zip(axes, ("CT1", "CT2", "CT3"),
                           ("(a) CT-1", "(b) CT-2", "(c) CT-3")):
        fs = sorted(glob.glob(os.path.join(
            RES, "w", f"gcnsac_tskf_pia_{ct}_gf180_s0_tskfhist.npy")))
        if not fs:
            continue
        H = np.load(fs[0])
        t = np.arange(len(H))
        ax.plot(t, np.abs(H[:, 0]), color=C[1], lw=0.9, label="|shaping error|")
        ax.plot(t, H[:, 1], color=C[0], lw=0.9, label="consequent mass")
        ax.plot(t, H[:, 2], color=C[2], lw=0.9, label="mean firing")
        ax.set_xlabel("training step")
        ax.set_title(ttl, fontsize=8)
    axes[0].set_ylabel("magnitude")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, loc="outside upper center", fontsize=7)
    fig.savefig(os.path.join(FIG, "FS1_tskf_dynamics.pdf"), bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    fig_learning()
    fig_tskf()
    fig_transfer()
    fig_convergence()
    fig_tskf_supp()
    try:
        fig_ablation()
    except Exception as e:
        print("ablation fig skipped:", e)
    print("figures written to", FIG)
