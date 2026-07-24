"""Physics-in-the-loop differentiable surrogate with adjoint gradients.

A square-law/EKV-style analytic device model with parameters (kp = mu*Cox,
channel-length modulation lambda(L), threshold vth) calibrated per PDK by
least squares against ngspice DC sweeps of the real BSIM models. The
surrogate maps the continuous action vector directly to an analytic figure
of merit through the corrected small-signal expressions of Table II. Because
the mapping is implemented in PyTorch, reverse-mode automatic
differentiation supplies the exact adjoint gradient d FoM_phys / d a, which
is injected into the actor update (adjoint-guided policy learning).
"""
import os, json, subprocess, tempfile, uuid
import numpy as np
import torch

CAL_DIR = os.path.expanduser("~/work/gcnsac/results/cal")

# ---------------- calibration against ngspice ----------------

def _dc_sweep(pdk, ptype, W, L):
    """Return (vgs, id) arrays at |VDS| = vdd/2 from a real ngspice DC sweep."""
    dev = pdk.mosfet(1, "d", "g", "0", "0", ptype, W, L)
    vdd = pdk.vdd
    sgn = -1.0 if ptype else 1.0
    net = "\n".join([
        "* cal", pdk.header(),
        f"VD d 0 {sgn*vdd/2:.3f}",
        "VG g 0 0",
        dev.replace(" vdd", " 0") if not ptype else dev,
        ".control",
        f"dc VG 0 {sgn*vdd:.3f} {sgn*vdd/40:.4f}",
        "print i(VD)",
        ".endc", ".end"])
    if ptype:  # source/bulk to vdd equivalent: shift by building explicit netlist
        net = "\n".join([
            "* cal p", pdk.header(),
            f"VDD vdd 0 {vdd}",
            f"VD d 0 {vdd/2:.3f}",
            "VG g 0 0",
            pdk.mosfet(1, "d", "g", "vdd", "vdd", 1, W, L),
            ".control",
            f"dc VG {vdd} 0 {-vdd/40:.4f}",
            "print i(VD)",
            ".endc", ".end"])
    f = os.path.join(tempfile.gettempdir(), f"cal_{uuid.uuid4().hex}.sp")
    open(f, "w").write(net)
    r = subprocess.run(["ngspice", "-b", f], capture_output=True, text=True, timeout=60)
    os.unlink(f)
    vg, iD = [], []
    on = False
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0].isdigit():
            try:
                vg.append(float(parts[1])); iD.append(abs(float(parts[2])))
                on = True
            except ValueError:
                pass
    return np.array(vg), np.array(iD)


def calibrate(pdk, n_sims_counter=None, force=False):
    """Fit (vth, kp, n_sub) for N and P and lambda(L) per device. ~24 DC sims."""
    os.makedirs(CAL_DIR, exist_ok=True)
    path = os.path.join(CAL_DIR, f"{pdk.name}.json")
    if os.path.exists(path) and not force:
        return json.load(open(path))
    out = {}
    Ls = [pdk.lmin, 2 * pdk.lmin, 4 * pdk.lmin]
    for ptype, key in [(0, "n"), (1, "p")]:
        W = 20 * pdk.wmin
        vg, iD = _dc_sweep(pdk, ptype, W, Ls[1])
        if n_sims_counter: n_sims_counter.bump()
        vgs = np.abs(vg) if not ptype else (pdk.vdd - vg)
        mask = iD > iD.max() * 1e-3
        # square-law fit in saturation: sqrt(ID) = sqrt(kp/2 * W/L) (VGS - vth)
        s = np.sqrt(iD[mask]); v = vgs[mask]
        A = np.vstack([v, np.ones_like(v)]).T
        m, b = np.linalg.lstsq(A, s, rcond=None)[0]
        vth = -b / m
        kp = 2 * m * m / (W / Ls[1])
        # EKV all-region fit: slope factor n by least squares of numerical gm
        # against the EKV interpolation gm = ID/(n*Ut) * 2/(1+sqrt(1+4*IC)),
        # IC = ID / (2*n*kp*(W/L)*Ut^2)
        Ut = 0.0258
        gmn = np.gradient(iD, vgs) if len(iD) > 5 else None
        best_n, best_err = 1.3, 1e18
        if gmn is not None:
            msk = iD > iD.max() * 1e-4
            for n_try in np.linspace(1.05, 2.0, 39):
                I0 = 2 * n_try * kp * (W / Ls[1]) * Ut * Ut
                IC = np.clip(iD[msk] / I0, 1e-9, None)
                gm_ekv = iD[msk] / (n_try * Ut) * 2.0 / (1 + np.sqrt(1 + 4 * IC))
                err = float(np.mean((gm_ekv - gmn[msk]) ** 2))
                if err < best_err:
                    best_err, best_n = err, n_try
        lam = {}
        for L in Ls:
            # lambda via ID(VDS) slope at fixed VGS
            vdd = pdk.vdd
            net = "\n".join([
                "* lam", pdk.header(),
                f"VDD vdd 0 {vdd}", "VG g 0 %.3f" % (vth + 0.35 * (vdd - vth)),
                "VD d 0 0",
                pdk.mosfet(1, "d", "g", "0", "0", 0, W, L) if not ptype else
                pdk.mosfet(1, "d", "g", "vdd", "vdd", 1, W, L),
                ".control",
                f"dc VD {0.2*vdd:.3f} {vdd:.3f} {vdd/20:.4f}" if not ptype else
                f"dc VD {0.8*vdd:.3f} 0 {-vdd/20:.4f}",
                "print i(VD)", ".endc", ".end"])
            if ptype:
                net = net.replace("VG g 0 %.3f" % (vth + 0.35 * (vdd - vth)),
                                  "VG g 0 %.3f" % (vdd - vth - 0.35 * (vdd - vth)))
            fpath = os.path.join(tempfile.gettempdir(), f"lam_{uuid.uuid4().hex}.sp")
            open(fpath, "w").write(net)
            r = subprocess.run(["ngspice", "-b", fpath], capture_output=True,
                               text=True, timeout=60)
            os.unlink(fpath)
            if n_sims_counter: n_sims_counter.bump()
            vd, iDD = [], []
            for line in r.stdout.splitlines():
                parts = line.split()
                if len(parts) == 3 and parts[0].isdigit():
                    try:
                        vd.append(float(parts[1])); iDD.append(abs(float(parts[2])))
                    except ValueError:
                        pass
            vd, iDD = np.array(vd), np.array(iDD)
            if len(vd) > 4 and iDD.max() > 0:
                half = len(vd) // 2
                p = np.polyfit(np.abs(vd[half:]), iDD[half:], 1)
                lam[f"{L:.3e}"] = float(max(p[0] / max(p[1], 1e-12), 1e-3))
            else:
                lam[f"{L:.3e}"] = 0.05
        out[key] = {"vth": float(abs(vth)), "kp": float(abs(kp)), "lam": lam,
                    "n_ekv": float(best_n)}
    json.dump(out, open(path, "w"))
    return out


# ---------------- differentiable surrogate ----------------

class PhysicsSurrogate:
    """Differentiable FoM proxy. Action tensor a (requires_grad) -> scalar.
    model='sq' uses the square-law gm; model='ekv' uses the all-region EKV
    interpolation, which stays accurate in moderate inversion and at low
    supply voltages."""

    def __init__(self, env, cal, model="sq"):
        self.env = env
        self.topo = env.topo
        self.pdk = env.pdk
        self.cal = cal
        self.model = model

    def _decode_t(self, a):
        """Torch version of Topology.decode (no grid snap; smooth)."""
        out = {}
        for i, (nm, kind, lo, hi, log) in enumerate(self.topo.params):
            t = torch.clamp((a[i] + 1.0) / 2.0, 0.0, 1.0)
            if log:
                v = lo * (hi / lo) ** t
            else:
                v = lo + t * (hi - lo)
            out[nm] = v
        return out

    def _gm(self, key, W, L, ID):
        kp = self.cal[key]["kp"]
        if self.model == "ekv":
            n = self.cal[key].get("n_ekv", 1.3)
            Ut = 0.0258
            I0 = 2.0 * n * kp * (W / L) * Ut * Ut
            IC = torch.clamp(ID / torch.clamp(I0, min=1e-18), min=1e-9)
            return ID / (n * Ut) * 2.0 / (1.0 + torch.sqrt(1.0 + 4.0 * IC))
        return torch.sqrt(torch.clamp(2.0 * kp * (W / L) * ID, min=1e-18))

    def _ro(self, key, L, ID):
        lams = self.cal[key]["lam"]
        Ls = sorted(float(k) for k in lams)
        lam_vals = [lams[f"{k:.3e}"] for k in Ls]
        # interpolate lambda in L (piecewise linear, clamped)
        Lc = torch.clamp(L, Ls[0], Ls[-1])
        lam = torch.ones_like(L) * lam_vals[-1]
        for j in range(len(Ls) - 1):
            m = (lam_vals[j + 1] - lam_vals[j]) / (Ls[j + 1] - Ls[j])
            seg = lam_vals[j] + m * (Lc - Ls[j])
            lam = torch.where((Lc >= Ls[j]) & (Lc <= Ls[j + 1]), seg, lam)
        return 1.0 / torch.clamp(lam * ID, min=1e-12)

    def fom_phys(self, a):
        """Analytic normalized FoM (corrected Table II expressions)."""
        p = self._decode_t(a)
        t = self.topo.name
        if t == "CT1":
            ID1 = p["IB"] / 2.0
            ID2 = p["IB"]
            gm1 = self._gm("n", p["W12"], p["L12"], ID1)
            gm6 = self._gm("p", p["W6"], p["L67"], ID2)
            ro2 = 1.0 / (1.0 / self._ro("n", p["L12"], ID1) +
                         1.0 / self._ro("p", p["L34"], ID1))
            ro6 = 1.0 / (1.0 / self._ro("p", p["L67"], ID2) +
                         1.0 / self._ro("n", p["L67"], ID2))
            av = gm1 * ro2 * gm6 * ro6
            ugf = gm1 / (2 * np.pi * p["CC"])            # unity-gain frequency
            CL = torch.tensor(1e-12)
            p2 = gm6 / (2 * np.pi * CL)                  # non-dominant pole
            pm = 90.0 - torch.rad2deg(torch.atan(ugf / p2))  # two-pole PM
            power = self.pdk.vdd * (2 * ID1 + ID2 + p["IB"])
            feats = {"av_db": 20 * torch.log10(torch.clamp(av, min=1e-3)),
                     "ugf": torch.log10(torch.clamp(ugf, min=1.0)),
                     "pm": pm, "power": torch.log10(power)}
        else:
            ns = self.topo.n_stage
            IDd = 0.2 * p["IB"]
            gmd = self._gm("n", p["WD"], p["LD"], IDd)
            zt = 1.0 / gmd
            for s in range(1, ns + 1):
                gms = self._gm("n", p[f"WN{s}"], p[f"L{s}"], p["IB"])
                ros = 1.0 / (1.0 / self._ro("n", p[f"L{s}"], p["IB"]) +
                             1.0 / self._ro("p", p[f"L{s}"], p["IB"]))
                zt = zt * gms * ros
            gm2 = self._gm("n", p[f"WN{ns}"], p[f"L{ns}"], p["IB"])
            # dominant pole at the compensation node
            ro1 = 1.0 / (1.0 / self._ro("n", p["L1"], p["IB"]) +
                         1.0 / self._ro("p", p["L1"], p["IB"]))
            f3db = 1.0 / (2 * np.pi * ro1 * torch.clamp(p["CC"], min=1e-15) *
                          torch.clamp(gm2 * ro1, min=1.0) ** 0.5)
            power = self.pdk.vdd * ((ns + 1.2) * p["IB"] + IDd)
            feats = {"zt": torch.log10(torch.clamp(zt, min=1e-3)),
                     "f3db": torch.log10(torch.clamp(f3db, min=1.0)),
                     "power": torch.log10(power)}
        # normalize with the environment's min-max stats and combine
        s = 0.0
        for m, v in feats.items():
            if m not in self.env.norm:
                continue
            lo, hi = self.env.norm[m]
            x = torch.clamp((v - lo) / (hi - lo), -0.5, 1.25)
            d = self.topo.directions[m]
            s = s + (x if d > 0 else (1.0 - x))
        return s

    def adjoint_grad(self, a_np):
        """Exact reverse-mode gradient of FoM_phys wrt the action vector."""
        a = torch.tensor(np.asarray(a_np, dtype=np.float64), requires_grad=True)
        f = self.fom_phys(a)
        f.backward()
        g = a.grad.detach().numpy()
        g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        n = np.linalg.norm(g)
        return g / n if n > 1e-12 else g
