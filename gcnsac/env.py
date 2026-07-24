"""ngspice-in-the-loop sizing environment with a unified, method-independent FoM.

The FoM (Eq. 2 of the paper) is computed by the ENVIRONMENT identically for
every optimizer: FoM = sum_i k_i * min(f_i, f_i^bound) normalized by the
min-max range collected from random sampling. Training rewards (standard or
TSKF-shaped) never change this evaluation metric.
"""
import os, re, subprocess, tempfile, uuid, json, math
import numpy as np
from .circuits import get_topology
from .pdk import get_pdk

_MEAS = re.compile(r"^(\w+)\s*=\s*([-+0-9.eE]+)", re.M)
_PRINTVAL = re.compile(r"^(pwr|inoise_total)\s*=?\s*([-+0-9.eE]+)", re.M)

class SimCounter:
    def __init__(self):
        self.n = 0
    def bump(self, k=1):
        self.n += k

class CircuitEnv:
    def __init__(self, topo_name, pdk_name, workdir=None, counter=None):
        self.pdk = get_pdk(pdk_name)
        self.topo = get_topology(topo_name, self.pdk)
        self.workdir = workdir or tempfile.gettempdir()
        self.counter = counter or SimCounter()
        self.norm = None      # per-metric (lo, hi) from random sampling
        self.bounds = None    # per-metric upper bound (anti over-exploration)
        self.weights = None   # per-metric importance |k_i| (default 1)

    # ---------------- simulation ----------------
    def simulate(self, action):
        """action in [-1,1]^d -> metrics dict (np.nan on failure)."""
        p = self.topo.decode(np.asarray(action, dtype=float))
        net = self.topo.build(p)
        f = os.path.join(self.workdir, f"gcnsac_{uuid.uuid4().hex}.sp")
        with open(f, "w") as fh:
            fh.write(net)
        try:
            r = subprocess.run(["ngspice", "-b", f], capture_output=True,
                               text=True, timeout=10)
            out = r.stdout
        except subprocess.TimeoutExpired:
            out = ""
        finally:
            os.unlink(f)
        self.counter.bump()
        vals = {}
        for m in list(_MEAS.finditer(out)) + list(_PRINTVAL.finditer(out)):
            try:
                vals[m.group(1).lower()] = float(m.group(2))
            except ValueError:
                pass
        met = {}
        for name in self.topo.metrics:
            key = {"av_db": "av_db", "ugf": "ugf", "pm": "phi",
                   "power": "pwr", "inoise": "inoise_total",
                   "zt": "zt", "f3db": "f3db"}[name]
            v = vals.get(key, np.nan)
            met[name] = v
        # post-processing
        if "pm" in met and np.isfinite(met["pm"]):
            # ngspice vp() is in radians. PM = 180 - phase drop from DC to UGF.
            phi0 = vals.get("phi0", math.pi)
            drop = math.degrees(abs(phi0 - met["pm"]))
            pm = 180.0 - (drop % 360.0)
            met["pm"] = float(np.clip(pm, -180.0, 180.0))
        if "power" in met and np.isfinite(met["power"]):
            met["power"] = abs(met["power"])
        if "av_db" in met and np.isfinite(met.get("av_db", np.nan)) and met["av_db"] < 0:
            met["_dead"] = True
        return met, p

    # ---------------- normalization ----------------
    def calibrate(self, n_samples=400, seed=0, save=None):
        """Random sampling to set min-max normalization. Counted in the budget."""
        rng = np.random.default_rng(seed)
        rows = []
        for _ in range(n_samples):
            a = rng.uniform(-1, 1, self.topo.dim)
            met, _ = self.simulate(a)
            rows.append(met)
        self.set_norm_from(rows)
        if save:
            with open(save, "w") as fh:
                json.dump({"norm": self.norm, "bounds": self.bounds}, fh)
        return rows

    def set_norm_from(self, rows):
        self.norm, self.bounds = {}, {}
        for m in self.topo.metrics:
            v = np.array([r.get(m, np.nan) for r in rows], float)
            v = v[np.isfinite(v)]
            if m in ("power", "inoise"):
                v = np.log10(np.clip(np.abs(v), 1e-30, None))
            elif m in ("ugf", "f3db", "zt"):
                v = np.log10(np.clip(np.abs(v), 1e-3, None))
            if len(v) < 5:
                self.norm[m] = (0.0, 1.0); self.bounds[m] = 1e30; continue
            lo, hi = np.percentile(v, 2), np.percentile(v, 98)
            self.norm[m] = (float(lo), float(hi) if hi > lo else float(lo) + 1)
            self.bounds[m] = float(hi + 0.25 * (hi - lo))

    def load_norm(self, path):
        with open(path) as fh:
            d = json.load(fh)
        self.norm, self.bounds = d["norm"], d["bounds"]

    def normalized(self, met):
        """Per-metric normalized values in [0, ~1.25], NaN-safe."""
        out = {}
        for m in self.topo.metrics:
            v = met.get(m, np.nan)
            if not np.isfinite(v):
                out[m] = 0.0 if self.topo.directions[m] > 0 else 1.5
                continue
            if m in ("power", "inoise"):
                v = math.log10(max(abs(v), 1e-30))
            elif m in ("ugf", "f3db", "zt"):
                v = math.log10(max(abs(v), 1e-3))
            elif m == "pm":
                v = float(np.clip(v, -90.0, 120.0))
            lo, hi = self.norm[m]
            x = (min(v, self.bounds[m]) - lo) / (hi - lo)
            out[m] = float(np.clip(x, -0.5, 1.25))
        return out

    def fom(self, met):
        """Unified environment FoM: identical for every optimizer."""
        if met.get("_dead"):
            return 0.0
        nz = self.normalized(met)
        s = 0.0
        for m in self.topo.metrics:
            k = 1.0 if self.weights is None else self.weights.get(m, 1.0)
            d = self.topo.directions[m]
            s += k * (nz[m] if d > 0 else (1.0 - nz[m]))
        if not any(np.isfinite(met.get(m, np.nan)) for m in self.topo.metrics):
            return 0.0
        return float(max(s, 0.0))

    def step(self, action):
        met, p = self.simulate(action)
        return met, p, self.fom(met)
