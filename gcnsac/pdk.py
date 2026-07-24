"""PDK abstraction for open-source technology nodes.

Nodes:
  gf180  : GlobalFoundries GF180MCU, 180 nm, open PDK (Apache-2.0), 3.3 V devices
  sky130 : SkyWater SKY130, 130 nm, open PDK (Apache-2.0), 1.8 V devices
  ptm65  : ASU Predictive Technology Model, 65 nm bulk CMOS, 1.1 V
  ptm45  : ASU Predictive Technology Model, 45 nm metal-gate/high-k, 1.0 V

All model files are open source and redistributable; no NDA material is used.
"""
import os

PDK_ROOT = os.environ.get("GCNSAC_PDK_ROOT", os.path.expanduser("~/work/pdk"))

class Pdk:
    def __init__(self, name, includes, nmos, pmos, vdd, use_subckt,
                 scale_option=None, wl_unit=1.0, lmin=None, lmax=None,
                 wmin=None, wmax=None, grid=5e-9):
        self.name = name
        self.includes = includes          # list of ngspice include/lib lines
        self.nmos = nmos                  # device/model name
        self.pmos = pmos
        self.vdd = vdd
        self.use_subckt = use_subckt      # X-device (subckt) vs M-device
        self.scale_option = scale_option  # e.g. ".option scale=1.0u"
        self.wl_unit = wl_unit            # multiply meters by this before printing
        self.lmin, self.lmax = lmin, lmax # bounds in meters
        self.wmin, self.wmax = wmin, wmax
        self.grid = grid                  # sizing grid in meters

    def header(self):
        lines = list(self.includes)
        if self.scale_option:
            lines.append(self.scale_option)
        return "\n".join(lines)

    def fmt_wl(self, meters):
        return f"{meters*self.wl_unit:.6g}"

    def mosfet(self, idx, d, g, s, b, ptype, w, l, m=1):
        """Return a device line. w, l in meters."""
        model = self.pmos if ptype else self.nmos
        ws, ls = self.fmt_wl(w), self.fmt_wl(l)
        if self.use_subckt:
            return f"XM{idx} {d} {g} {s} {b} {model} w={ws} l={ls} nf=1 m={m}"
        return f"M{idx} {d} {g} {s} {b} {model} w={ws} l={ls} m={m}"


def get_pdk(name):
    if name == "gf180":
        return Pdk(
            "gf180",
            [f".include {PDK_ROOT}/gf180/models/ngspice/design.ngspice",
             f".lib {PDK_ROOT}/gf180/models/ngspice/sm141064.ngspice typical"],
            "nmos_3p3", "pmos_3p3", 3.3, use_subckt=True, wl_unit=1.0,
            lmin=0.28e-6, lmax=4e-6, wmin=0.22e-6, wmax=200e-6, grid=10e-9)
    if name == "sky130":
        return Pdk(
            "sky130",
            [f".include {PDK_ROOT}/sky130/models/mini_tt.spice"],
            "sky130_fd_pr__nfet_01v8", "sky130_fd_pr__pfet_01v8", 1.8,
            use_subckt=True, scale_option=".option scale=1.0u", wl_unit=1e6,
            lmin=0.15e-6, lmax=4e-6, wmin=0.42e-6, wmax=100e-6, grid=5e-9)
    if name == "ptm65":
        return Pdk(
            "ptm65",
            [f".include {PDK_ROOT}/ptm/65nm_bulk.pm"],
            "nmos", "pmos", 1.1, use_subckt=False, wl_unit=1.0,
            lmin=65e-9, lmax=1e-6, wmin=0.13e-6, wmax=50e-6, grid=5e-9)
    if name == "ptm45":
        return Pdk(
            "ptm45",
            [f".include {PDK_ROOT}/ptm/45nm_bulk.pm"],
            "nmos", "pmos", 1.0, use_subckt=False, wl_unit=1.0,
            lmin=45e-9, lmax=1e-6, wmin=0.09e-6, wmax=50e-6, grid=5e-9)
    raise ValueError(name)
