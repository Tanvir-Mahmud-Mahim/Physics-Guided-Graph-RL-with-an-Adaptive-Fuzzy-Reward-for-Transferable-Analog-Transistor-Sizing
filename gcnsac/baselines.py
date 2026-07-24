"""Classical black-box baselines under identical simulator-evaluation budgets:
BO (GP expected improvement) and MACE-style multi-acquisition ensemble."""
import numpy as np

class _GP:
    """Lightweight GP with RBF kernel (no sklearn dependency)."""
    def __init__(self, ls=0.6, sf=1.0, sn=1e-2):
        self.ls, self.sf, self.sn = ls, sf, sn
        self.X, self.y = None, None

    def _k(self, A, B):
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return self.sf * np.exp(-0.5 * d2 / self.ls ** 2)

    def fit(self, X, y):
        self.X = np.asarray(X, float)
        self.mu0 = float(np.mean(y))
        self.y = np.asarray(y, float) - self.mu0
        K = self._k(self.X, self.X) + self.sn * np.eye(len(y))
        self.L = np.linalg.cholesky(K + 1e-8 * np.eye(len(y)))
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y))

    def predict(self, Xs):
        Ks = self._k(np.asarray(Xs, float), self.X)
        mu = Ks @ self.alpha + self.mu0
        v = np.linalg.solve(self.L, Ks.T)
        var = np.clip(self.sf - (v ** 2).sum(0), 1e-9, None)
        return mu, np.sqrt(var)


def _norm_cdf(x):
    from math import erf, sqrt
    return 0.5 * (1 + np.vectorize(erf)(x / np.sqrt(2)))

def _norm_pdf(x):
    return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)


def run_bo(env, budget, seed=0, n_init=40, mace=False, log_every=1, cb=None):
    """Sequential (BO) or 3-point multi-acquisition (MACE-style) optimizer.
    Returns history list of (step, fom)."""
    rng = np.random.default_rng(seed)
    d = env.topo.dim
    X, y, hist = [], [], []
    best = -1e9
    for i in range(n_init):
        a = rng.uniform(-1, 1, d)
        _, _, f = env.step(a)
        X.append(a); y.append(f)
        best = max(best, f)
        hist.append((i + 1, best))
        if cb: cb(i + 1, best)
    gp = _GP(ls=0.6 * np.sqrt(d))
    n = n_init
    refit_every = 5
    since_fit = refit_every
    while n < budget:
        if since_fit >= refit_every:
            gp.fit(np.array(X), np.array(y))
            since_fit = 0
        cand = rng.uniform(-1, 1, (600, d))
        mu, s = gp.predict(cand)
        fbest = max(y)
        z = (mu - fbest) / s
        ei = (mu - fbest) * _norm_cdf(z) + s * _norm_pdf(z)
        if mace:
            ucb = mu + 1.8 * s
            pi = _norm_cdf(z)
            picks = [cand[int(np.argmax(acq))] for acq in (ei, ucb, pi)]
        else:
            picks = [cand[int(np.argmax(ei))]]
        for a in picks:
            if n >= budget:
                break
            _, _, f = env.step(a)
            X.append(a); y.append(f)
            n += 1
            since_fit += 1
            best = max(best, f)
            hist.append((n, best))
            if cb: cb(n, best)
    return hist, (np.array(X), np.array(y))
