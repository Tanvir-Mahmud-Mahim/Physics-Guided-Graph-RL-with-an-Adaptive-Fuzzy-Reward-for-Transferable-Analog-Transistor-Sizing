"""Parameterized testbed circuit topologies.

CT-1: two-stage Miller-compensated voltage amplifier (5T first stage + CS second stage)
CT-2: two-stage transimpedance amplifier (diode-connected input + 2 gain stages)
CT-3: three-stage transimpedance amplifier (diode-connected input + 3 gain stages)

Each topology defines:
  params  : list of (name, kind, lo, hi, log) design variables (meters, F, A)
  build() : ngspice netlist for a given sizing vector
  graph() : (node_feats_meta, edge_index) circuit graph, nodes = devices
  metrics : names of measured performance features
"""
import numpy as np

def _snap(x, grid):
    return max(grid, round(x / grid) * grid)

class Topology:
    name = ""
    metrics = []
    # metric direction: +1 more is better, -1 less is better (for env FoM)
    directions = {}

    def __init__(self, pdk):
        self.pdk = pdk
        self.params = self._params()
        self.dim = len(self.params)

    def _params(self):
        raise NotImplementedError

    def decode(self, a):
        """Map action in [-1,1]^d to physical parameter dict (grid-snapped)."""
        out = {}
        for ai, (nm, kind, lo, hi, log) in zip(a, self.params):
            t = float(np.clip((ai + 1.0) / 2.0, 0.0, 1.0))
            v = lo * (hi / lo) ** t if log else lo + t * (hi - lo)
            if kind in ("W", "L"):
                v = _snap(v, self.pdk.grid)
            out[nm] = v
        return out

    def encode_feats(self, p):
        """Per-device normalized numeric features for graph nodes."""
        raise NotImplementedError


class CT1(Topology):
    """Two-stage Miller OTA. Devices:
    M1,M2 diff pair (N), M3,M4 mirror load (P), M5 tail (N), M8 bias diode (N),
    M6 CS driver (P), M7 CS load (N), Cc Miller cap, CL fixed, IREF fixed."""
    name = "CT1"
    metrics = ["av_db", "ugf", "pm", "power", "inoise"]
    directions = {"av_db": +1, "ugf": +1, "pm": +1, "power": -1, "inoise": -1}

    def _params(self):
        k = self.pdk
        return [
            ("W12", "W", 2 * k.wmin, k.wmax, True),
            ("L12", "L", k.lmin, k.lmax, True),
            ("W34", "W", 2 * k.wmin, k.wmax, True),
            ("L34", "L", k.lmin, k.lmax, True),
            ("W5",  "W", 2 * k.wmin, k.wmax, True),
            ("W6",  "W", 2 * k.wmin, k.wmax, True),
            ("L67", "L", k.lmin, k.lmax, True),
            ("W7",  "W", 2 * k.wmin, k.wmax, True),
            ("CC",  "C", 0.1e-12, 10e-12, True),
            ("IB",  "I", 1e-6, 50e-6, True),
        ]

    def build(self, p):
        k = self.pdk
        vdd = k.vdd
        vcm = 0.45 * vdd if k.name in ("ptm65", "ptm45") else 0.42 * vdd
        n = []
        n.append("* CT1 two-stage Miller OTA  (%s)" % k.name)
        n.append(k.header())
        n.append(f"VDD vdd 0 {vdd}")
        n.append(f"VINP inp 0 dc {vcm:.4f} ac 0.5")
        n.append(f"VINN inn 0 dc {vcm:.4f} ac -0.5")
        n.append(f"IREF vdd nb dc {p['IB']:.4e}")
        n.append(k.mosfet(8, "nb", "nb", "0", "0", 0, p["W5"], p["L12"]))     # bias diode
        n.append(k.mosfet(5, "ntail", "nb", "0", "0", 0, p["W5"], p["L12"]))  # tail
        n.append(k.mosfet(1, "n1", "inp", "ntail", "0", 0, p["W12"], p["L12"]))
        n.append(k.mosfet(2, "no1", "inn", "ntail", "0", 0, p["W12"], p["L12"]))
        n.append(k.mosfet(3, "n1", "n1", "vdd", "vdd", 1, p["W34"], p["L34"]))
        n.append(k.mosfet(4, "no1", "n1", "vdd", "vdd", 1, p["W34"], p["L34"]))
        n.append(k.mosfet(6, "out", "no1", "vdd", "vdd", 1, p["W6"], p["L67"]))
        n.append(k.mosfet(7, "out", "nb", "0", "0", 0, p["W7"], p["L67"]))
        n.append(f"CC no1 out {p['CC']:.4e}")
        n.append("CL out 0 1p")
        n.append(".control")
        n.append("op")
        n.append("let pwr = -v(vdd)*i(VDD)")
        n.append("print pwr")
        n.append("ac dec 20 1 100G")
        n.append("meas ac av_db FIND vdb(out) AT=10")
        n.append("meas ac ugf WHEN vdb(out)=0 CROSS=1")
        n.append("meas ac phi FIND vp(out) WHEN vdb(out)=0 CROSS=1")
        n.append("meas ac phi0 FIND vp(out) AT=10")
        n.append("noise v(out) VINP dec 10 10 10meg")
        n.append("print inoise_total")
        n.append(".endc")
        n.append(".end")
        return "\n".join(n)

    # graph: 0:M1 1:M2 2:M3 3:M4 4:M5 5:M6 6:M7 7:M8 8:CC 9:CL 10:IREF
    NODE_TYPES = [0, 0, 1, 1, 0, 1, 0, 0, 3, 3, 4]  # 0 nmos,1 pmos,2 res,3 cap,4 src
    EDGES = [(0, 1), (0, 2), (1, 3), (2, 3), (0, 4), (1, 4), (1, 5), (3, 5),
             (5, 6), (4, 7), (6, 7), (7, 10), (5, 8), (1, 8), (5, 9), (6, 9)]
    STAGE = [1, 1, 1, 1, 1, 2, 2, 0, 2, 2, 0]

    def node_params(self, p):
        return [("W12", "L12"), ("W12", "L12"), ("W34", "L34"), ("W34", "L34"),
                ("W5", "L12"), ("W6", "L67"), ("W7", "L67"), ("W5", "L12"),
                ("CC", None), (None, None), ("IB", None)]


class _TIA(Topology):
    """Shared TIA scaffold: photodiode-like current input, diode-connected NMOS
    I-V front end, then n_stage CS gain stages with active loads and Miller caps."""
    n_stage = 2
    metrics = ["zt", "f3db", "power", "inoise"]
    directions = {"zt": +1, "f3db": +1, "power": -1, "inoise": -1}

    def _params(self):
        k = self.pdk
        ps = [("WD", "W", k.wmin, 0.3 * k.wmax, True),
              ("LD", "L", k.lmin, k.lmax, True)]
        for s in range(1, self.n_stage + 1):
            ps += [(f"WN{s}", "W", 2 * k.wmin, k.wmax, True),
                   (f"WP{s}", "W", 2 * k.wmin, k.wmax, True),
                   (f"L{s}",  "L", k.lmin, k.lmax, True)]
        ps.append(("CC", "C", 0.05e-12, 5e-12, True))
        ps.append(("IB", "I", 1e-6, 100e-6, True))
        return ps

    def build(self, p):
        k = self.pdk
        vdd = k.vdd
        n = [f"* {self.name} {self.n_stage}-stage TIA ({k.name})", k.header()]
        n.append(f"VDD vdd 0 {vdd}")
        n.append(f"IIN 0 vx dc {0.2*p['IB']:.4e} ac 1")
        n.append("CPD vx 0 50f")
        n.append(k.mosfet(0, "vx", "vx", "0", "0", 0, p["WD"], p["LD"]))  # diode input
        n.append(f"IREF vdd nb dc {p['IB']:.4e}")
        n.append(k.mosfet(90, "nb", "nb", "0", "0", 0, p["WD"], p["LD"]))
        prev = "vx"
        for s in range(1, self.n_stage + 1):
            o = "out" if s == self.n_stage else f"o{s}"
            n.append(k.mosfet(2 * s, o, prev, "0", "0", 0, p[f"WN{s}"], p[f"L{s}"]))
            n.append(k.mosfet(2 * s + 1, o, "pb", "vdd", "vdd", 1, p[f"WP{s}"], p[f"L{s}"]))
            prev = o
        # PMOS bias rail
        n.append(k.mosfet(91, "pb", "pb", "vdd", "vdd", 1, p["WP1"], p["L1"]))
        n.append(f"IREFP pb 0 dc {p['IB']:.4e}")
        # Miller cap around the last stage
        pre_last = "vx" if self.n_stage == 1 else f"o{self.n_stage-1}"
        n.append(f"CC {pre_last} out {p['CC']:.4e}")
        n.append("CL out 0 100f")
        n.append(".control")
        n.append("op")
        n.append("let pwr = -v(vdd)*i(VDD)")
        n.append("print pwr")
        n.append("ac dec 20 1k 100G")
        n.append("let ztf = vm(out)")
        n.append("meas ac zt FIND vm(out) AT=10k")
        n.append("meas ac ztdb FIND vdb(out) AT=10k")
        n.append("let target = ztdb-3")
        n.append("meas ac f3db WHEN vdb(out)=target CROSS=1")
        n.append("noise v(out) IIN dec 10 1k 1g")
        n.append("print inoise_total")
        n.append(".endc")
        n.append(".end")
        return "\n".join(n)

    @property
    def NODE_TYPES(self):
        # MD, M90(bias), per stage (MN, MP), M91(pbias), CC, CL, CPD, IIN, IREF
        t = [0, 0]
        for _ in range(self.n_stage):
            t += [0, 1]
        t += [1, 3, 3, 3, 4, 4]
        return t

    @property
    def EDGES(self):
        e = [(0, 1)]                 # MD gate/bias relation
        base = 2
        prev_dev = 0
        for s in range(self.n_stage):
            mn, mp = base + 2 * s, base + 2 * s + 1
            e += [(prev_dev, mn), (mn, mp)]
            prev_dev = mn
        pb = base + 2 * self.n_stage
        for s in range(self.n_stage):
            e.append((base + 2 * s + 1, pb))
        cc, cl, cpd, iin, iref = pb + 1, pb + 2, pb + 3, pb + 4, pb + 5
        e += [(prev_dev, cc), (prev_dev, cl), (0, cpd), (0, iin), (1, iref)]
        return e

    @property
    def STAGE(self):
        t = [0, 0]
        for s in range(self.n_stage):
            t += [s + 1, s + 1]
        t += [0, self.n_stage, self.n_stage, 0, 0, 0]
        return t

    def node_params(self, p):
        np_ = [("WD", "LD"), ("WD", "LD")]
        for s in range(1, self.n_stage + 1):
            np_ += [(f"WN{s}", f"L{s}"), (f"WP{s}", f"L{s}")]
        np_ += [("WP1", "L1"), ("CC", None), (None, None), (None, None),
                (None, None), ("IB", None)]
        return np_


class CT2(_TIA):
    name = "CT2"
    n_stage = 2

class CT3(_TIA):
    name = "CT3"
    n_stage = 3


def get_topology(name, pdk):
    return {"CT1": CT1, "CT2": CT2, "CT3": CT3}[name](pdk)
