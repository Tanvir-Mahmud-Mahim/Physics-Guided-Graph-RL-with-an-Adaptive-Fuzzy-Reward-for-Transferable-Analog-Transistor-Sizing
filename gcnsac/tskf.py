"""Adaptive interval type-2 Takagi-Sugeno-Kang fuzzy (IT2-TSKF) reward.

Genuine interval type-2 implementation:
  - each fuzzy set has lower/upper membership functions separated by a
    footprint of uncertainty (FOU) of width delta > 0,
  - rule firing strengths are intervals [w_lo, w_hi],
  - the crisp output is obtained with Karnik-Mendel (KM) type reduction,
  - rule consequents are linear functions of the signed features and are
    adapted online with a momentum gradient update and an adaptive
    learning rate (normalized-gradient step).

Setting delta=0 recovers the type-1 system used for the ablation study.
"""
import numpy as np
import itertools

class IT2TSKF:
    def __init__(self, n_features, delta=0.15, eta0=0.05, beta=0.9,
                 c_hi=1.02, c_lo=0.995, seed=0, type1=False):
        self.n = n_features
        self.delta = 0.0 if type1 else float(delta)
        self.eta0, self.beta = eta0, beta
        self.c_hi, self.c_lo = c_hi, c_lo
        rng = np.random.default_rng(seed)
        self.rules = list(itertools.product(range(3), repeat=self.n))  # 3^n rules
        self.R = len(self.rules)
        self.W = 0.1 * rng.standard_normal((self.R, self.n))  # consequents
        self.G = np.zeros_like(self.W)                         # momentum buffer
        # principal membership params per set: low / med / high on [0,1.25]
        self.centers = np.array([0.25, 0.7, 1.05])
        self.slope = 4.0
        self.hist = []                                          # adaptation trace

    # ---- memberships ----
    def _mu(self, f):
        """Principal memberships mu[set] for scalar f (piecewise linear)."""
        lo = np.clip(self.slope * (self.centers[0] + 0.25 - f), 0.0, 1.0)
        med = np.clip(1.0 - self.slope * abs(f - self.centers[1]) / 2.0, 0.0, 1.0)
        hi = np.clip(self.slope * (f - self.centers[2] + 0.25), 0.0, 1.0)
        return np.array([lo, med, hi])

    def _mu_interval(self, f):
        mu = self._mu(f)
        mu_lo = np.clip(mu * (1.0 - self.delta), 0.0, 1.0)
        mu_hi = np.clip(mu * (1.0 + self.delta) + (self.delta * 0.05), 0.0, 1.0)
        return mu_lo, mu_hi

    # ---- inference ----
    def forward(self, f_norm, signs):
        """f_norm: normalized features in [0,1.25]; signs: +-1 direction vector.
        Returns crisp reward via KM type reduction and caches for adaptation."""
        f = np.asarray(f_norm, float)
        s = np.asarray(signs, float)
        ft = f * s                                     # signed feature vector
        mulos, muhis = zip(*[self._mu_interval(x) for x in f])
        mulos, muhis = np.array(mulos), np.array(muhis)  # (n, 3)
        idx = np.array(self.rules)                       # (R, n)
        w_lo = np.prod(mulos[np.arange(self.n), idx], axis=1)
        w_hi = np.prod(muhis[np.arange(self.n), idx], axis=1)
        y = self.W @ ft                                  # rule outputs (R,)
        y_l = self._km(y, w_lo, w_hi, left=True)
        y_r = self._km(y, w_lo, w_hi, left=False)
        r = 0.5 * (y_l + y_r)
        self._cache = (ft, w_lo, w_hi, y, r)
        return float(r)

    @staticmethod
    def _km(y, w_lo, w_hi, left=True, iters=20):
        """Karnik-Mendel iterative type reduction (enhanced KM not needed at R<=243)."""
        order = np.argsort(y)
        ys, wl, wh = y[order], w_lo[order], w_hi[order]
        w = 0.5 * (wl + wh)
        denom = w.sum() + 1e-12
        yk = (w * ys).sum() / denom
        for _ in range(iters):
            k = np.searchsorted(ys, yk)
            if left:
                w = np.concatenate([wh[:k], wl[k:]])
            else:
                w = np.concatenate([wl[:k], wh[k:]])
            new = (w * ys).sum() / (w.sum() + 1e-12)
            if abs(new - yk) < 1e-10:
                yk = new; break
            yk = new
        return yk

    # ---- adaptation ----
    def adapt(self, f_tar_norm):
        """Momentum gradient step of consequents toward the soft target."""
        ft, w_lo, w_hi, y, r = self._cache
        e = f_tar_norm - r
        w = 0.5 * (w_lo + w_hi)
        wn = w / (w.sum() + 1e-12)
        grad = np.outer(wn * e, ft)                     # (R, n)
        self.G = self.beta * self.G + grad
        eta = self.eta0 / (1.0 + np.linalg.norm(self.G, axis=1, keepdims=True))
        self.W += eta * self.G
        # rule usefulness scaling: |y_k| compared with half of mean firing level
        thr = 0.5 * w.mean()
        scale = np.where(np.abs(y) * wn > thr / max(self.R, 1), self.c_hi, self.c_lo)
        self.W *= scale[:, None]
        self.hist.append((float(e), float(np.abs(self.W).mean()),
                          float(w.mean()), float(r)))
        return float(e)
