"""RL agents: GCN-SAC (proposed, with PER, augmentation-based conservative
learning, optional IT2-TSKF reward and adjoint guidance), GCN-DDPG (the
GCN-RL baseline of Wang et al., DAC 2020), and non-graph A2C / PPO."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = torch.device("cpu")
torch.set_num_threads(1)

# ---------------- graph utilities ----------------

def build_graph(topo, p, cal=None):
    """Return (X, A_hat) node features and normalized adjacency."""
    types = topo.NODE_TYPES
    stages = topo.STAGE
    npar = topo.node_params(p)
    N = len(types)
    X = np.zeros((N, 11), dtype=np.float32)
    k = topo.pdk
    for i in range(N):
        X[i, types[i]] = 1.0
        wname, lname = npar[i]
        if wname and wname.startswith("W"):
            w = p[wname]
            X[i, 5] = np.log(w / k.wmin) / np.log(k.wmax / k.wmin)
        if lname:
            X[i, 6] = np.log(p[lname] / k.lmin) / np.log(k.lmax / k.lmin)
        if wname and not wname.startswith("W"):   # C or I value
            X[i, 7] = np.log10(max(p[wname], 1e-15)) / 15.0 + 1.0
        X[i, 8] = stages[i] / 3.0
        if cal is not None and types[i] in (0, 1) and wname and wname.startswith("W"):
            key = "n" if types[i] == 0 else "p"
            kp = cal[key]["kp"]
            ID = p.get("IB", 1e-5)
            gm = np.sqrt(max(2 * kp * (p[wname] / p[lname or "L12"]) * ID, 1e-18)) \
                if lname else 0.0
            X[i, 9] = np.clip(np.log10(max(gm, 1e-9)) / 6.0 + 1.0, 0, 2)
            X[i, 10] = np.clip(ID * 1e5, 0, 2)
    edges = topo.EDGES
    A = np.eye(N, dtype=np.float32)
    for (u, v) in edges:
        A[u, v] = A[v, u] = 1.0
    d = A.sum(1)
    Dm = np.diag(1.0 / np.sqrt(d))
    return X, (Dm @ A @ Dm).astype(np.float32)


def augment(X, A, p_drop=0.1, p_mask=0.1, rng=None):
    rng = rng or np.random
    Xa = X.copy(); Aa = A.copy()
    N = X.shape[0]
    mask = rng.random(Xa[:, 5:].shape) < p_mask
    Xa[:, 5:] = np.where(mask, 0.0, Xa[:, 5:])
    idx = np.transpose(np.nonzero(np.triu(Aa, 1)))
    for (u, v) in idx:
        if rng.random() < p_drop:
            Aa[u, v] = Aa[v, u] = 0.0
    d = np.maximum(Aa.sum(1), 1e-6)
    Dm = np.diag(1.0 / np.sqrt(d))
    return Xa, (Dm @ (Aa + np.eye(N, dtype=Aa.dtype) * 0) @ Dm).astype(np.float32)


class GCNEncoder(nn.Module):
    def __init__(self, fin=11, hid=64, zdim=64, layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        d = fin
        for _ in range(layers):
            self.convs.append(nn.Linear(d, hid))
            d = hid
        self.proj = nn.Linear(hid, zdim)
        self.head = nn.Sequential(nn.Linear(zdim, zdim), nn.ReLU(),
                                  nn.Linear(zdim, zdim))  # SimCLR projection

    def forward(self, X, A):
        h = X
        for c in self.convs:
            h = F.relu(c(A @ h))
        z = self.proj(h.mean(dim=-2))
        return z

    def simclr_z(self, X, A):
        return self.head(self.forward(X, A))


def nt_xent(z1, z2, tau=0.2):
    z1 = F.normalize(z1, dim=-1); z2 = F.normalize(z2, dim=-1)
    B = z1.shape[0]
    z = torch.cat([z1, z2], 0)
    sim = z @ z.t() / tau
    sim.fill_diagonal_(-1e9)
    targets = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, targets)


# ---------------- prioritized replay ----------------

class PER:
    def __init__(self, cap=20000, alpha=0.6, beta=0.4, eps=1e-3):
        self.cap, self.alpha, self.beta, self.eps = cap, alpha, beta, eps
        self.data, self.prio = [], []

    def push(self, item, prio=1.0):
        self.data.append(item); self.prio.append(abs(prio) + self.eps)
        if len(self.data) > self.cap:
            self.data.pop(0); self.prio.pop(0)

    def sample(self, batch, rng):
        p = np.array(self.prio) ** self.alpha
        p /= p.sum()
        idx = rng.choice(len(self.data), size=min(batch, len(self.data)),
                         p=p, replace=False)
        w = (len(self.data) * p[idx]) ** (-self.beta)
        w /= w.max()
        return idx, [self.data[i] for i in idx], w.astype(np.float32)

    def update(self, idx, td):
        for i, t in zip(idx, td):
            self.prio[i] = abs(float(t)) + self.eps


# ---------------- actors / critics ----------------

class GaussianActor(nn.Module):
    def __init__(self, zdim, adim, hid=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(zdim, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU())
        self.mu = nn.Linear(hid, adim)
        self.logstd = nn.Linear(hid, adim)

    def forward(self, z):
        h = self.net(z)
        mu = self.mu(h)
        logstd = torch.clamp(self.logstd(h), -5.0, 1.0)
        return mu, logstd

    def sample(self, z):
        mu, logstd = self(z)
        std = logstd.exp()
        eps = torch.randn_like(mu)
        u = mu + std * eps
        a = torch.tanh(u)
        logp = (-0.5 * (((u - mu) / std) ** 2 + 2 * logstd + np.log(2 * np.pi))
                - torch.log(1 - a ** 2 + 1e-6)).sum(-1)
        return a, logp, torch.tanh(mu)


class QNet(nn.Module):
    def __init__(self, zdim, adim, hid=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(zdim + adim, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 1))

    def forward(self, z, a):
        return self.net(torch.cat([z, a], -1)).squeeze(-1)


# ---------------- GCN-SAC agent ----------------

class GCNSAC:
    def __init__(self, topo, cal, adim, seed=0, use_per=True, use_aug=True,
                 use_simclr=True, surrogate=None, lam_adj=0.3, gamma=0.5,
                 lr=3e-4, batch=64):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.topo, self.cal = topo, cal
        self.enc = GCNEncoder()
        self.actor = GaussianActor(64, adim)
        self.q1, self.q2 = QNet(64, adim), QNet(64, adim)
        self.q1t, self.q2t = QNet(64, adim), QNet(64, adim)
        self.q1t.load_state_dict(self.q1.state_dict())
        self.q2t.load_state_dict(self.q2.state_dict())
        self.log_alpha = torch.tensor(np.log(0.1), requires_grad=True)
        self.target_H = -adim
        self.opt_a = torch.optim.Adam(list(self.actor.parameters()) +
                                      list(self.enc.parameters()), lr=lr)
        self.opt_q = torch.optim.Adam(list(self.q1.parameters()) +
                                      list(self.q2.parameters()), lr=lr)
        self.opt_al = torch.optim.Adam([self.log_alpha], lr=lr)
        self.buf = PER() if use_per else None
        self.simple_buf = []
        self.use_per, self.use_aug, self.use_simclr = use_per, use_aug, use_simclr
        self.surrogate, self.lam_adj = surrogate, lam_adj
        self.gamma, self.batch = gamma, batch

    def _z(self, X, A):
        return self.enc(torch.tensor(X), torch.tensor(A))

    def act(self, p, explore=True):
        X, A = build_graph(self.topo, p, self.cal)
        with torch.no_grad():
            z = self._z(X, A)
            a, _, mu = self.actor.sample(z.unsqueeze(0))
        a = (a if explore else mu).squeeze(0).numpy()
        return np.clip(a, -1, 1), (X, A)

    def store(self, s, a, r, s2, done):
        item = (s, a, r, s2, done)
        if self.use_per:
            self.buf.push(item, prio=abs(r) + 1.0)
        else:
            self.simple_buf.append(item)
            if len(self.simple_buf) > 20000:
                self.simple_buf.pop(0)

    def _sample(self):
        if self.use_per:
            idx, items, w = self.buf.sample(self.batch, self.rng)
        else:
            n = len(self.simple_buf)
            idx = self.rng.choice(n, size=min(self.batch, n), replace=False)
            items = [self.simple_buf[i] for i in idx]
            w = np.ones(len(idx), dtype=np.float32)
        return idx, items, w

    def update(self, steps=1):
        if (len(self.buf.data) if self.use_per else len(self.simple_buf)) < self.batch:
            return {}
        logs = {}
        for _ in range(steps):
            idx, items, w = self._sample()
            Xs = torch.tensor(np.stack([it[0][0] for it in items]))
            As = torch.tensor(np.stack([it[0][1] for it in items]))
            a = torch.tensor(np.stack([it[1] for it in items]), dtype=torch.float32)
            r = torch.tensor([it[2] for it in items], dtype=torch.float32)
            X2 = torch.tensor(np.stack([it[3][0] for it in items]))
            A2 = torch.tensor(np.stack([it[3][1] for it in items]))
            d = torch.tensor([float(it[4]) for it in items])
            wt = torch.tensor(w)
            z = self.enc(Xs, As)
            with torch.no_grad():
                z2 = self.enc(X2, A2)
                a2, logp2, _ = self.actor.sample(z2)
                alpha = self.log_alpha.exp()
                qt = torch.min(self.q1t(z2, a2), self.q2t(z2, a2))
                y = r + self.gamma * (1 - d) * (qt - alpha * logp2)
            q1, q2 = self.q1(z.detach(), a), self.q2(z.detach(), a)
            td = (q1 - y).detach().numpy()
            lq = (wt * ((q1 - y) ** 2 + (q2 - y) ** 2)).mean()
            self.opt_q.zero_grad(); lq.backward(); self.opt_q.step()
            if self.use_per:
                self.buf.update(idx, td)
            # actor + encoder (+ SimCLR + adjoint guidance)
            a_pi, logp, _ = self.actor.sample(z)
            qpi = torch.min(self.q1(z, a_pi), self.q2(z, a_pi))
            la = (self.log_alpha.exp().detach() * logp - qpi).mean()
            if self.use_simclr:
                views = [augment(it[0][0], it[0][1], rng=self.rng)
                         if self.use_aug else (it[0][0], it[0][1]) for it in items]
                views2 = [augment(it[0][0], it[0][1], rng=self.rng)
                          if self.use_aug else (it[0][0], it[0][1]) for it in items]
                zv1 = self.enc.simclr_z(torch.tensor(np.stack([v[0] for v in views])),
                                        torch.tensor(np.stack([v[1] for v in views])))
                zv2 = self.enc.simclr_z(torch.tensor(np.stack([v[0] for v in views2])),
                                        torch.tensor(np.stack([v[1] for v in views2])))
                la = la + 0.1 * nt_xent(zv1, zv2)
            if self.surrogate is not None:
                # adjoint guidance: push the deterministic policy along the
                # physics-surrogate gradient (differentiable analytic FoM)
                mu, _ = self.actor(z)
                a_mu = torch.tanh(mu)
                f_phys = torch.stack([self.surrogate.fom_phys(a_mu[i].double())
                                      for i in range(min(8, a_mu.shape[0]))])
                la = la - self.lam_adj * f_phys.float().mean()
            self.opt_a.zero_grad(); la.backward(); self.opt_a.step()
            lal = -(self.log_alpha * (logp.detach() + self.target_H)).mean()
            self.opt_al.zero_grad(); lal.backward(); self.opt_al.step()
            with torch.no_grad():
                for tp, sp in [(self.q1t, self.q1), (self.q2t, self.q2)]:
                    for pt, ps in zip(tp.parameters(), sp.parameters()):
                        pt.mul_(0.995).add_(0.005 * ps)
            logs = {"lq": float(lq), "la": float(la),
                    "alpha": float(self.log_alpha.exp())}
        return logs


# ---------------- GCN-DDPG (GCN-RL baseline) ----------------

class GCNDDPG:
    def __init__(self, topo, cal, adim, seed=0, gamma=0.5, lr=3e-4, batch=64):
        torch.manual_seed(seed + 77)
        self.rng = np.random.default_rng(seed + 77)
        self.topo, self.cal = topo, cal
        self.enc = GCNEncoder()
        self.actor = nn.Sequential(nn.Linear(64, 128), nn.ReLU(),
                                   nn.Linear(128, 128), nn.ReLU(),
                                   nn.Linear(128, adim), nn.Tanh())
        self.q = QNet(64, adim)
        self.qt = QNet(64, adim)
        self.qt.load_state_dict(self.q.state_dict())
        self.opt_a = torch.optim.Adam(list(self.actor.parameters()) +
                                      list(self.enc.parameters()), lr=lr)
        self.opt_q = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buf = []
        self.gamma, self.batch = gamma, batch
        self.sigma = 0.4

    def act(self, p, explore=True):
        X, A = build_graph(self.topo, p, self.cal)
        with torch.no_grad():
            z = self.enc(torch.tensor(X), torch.tensor(A))
            a = self.actor(z.unsqueeze(0)).squeeze(0).numpy()
        if explore:
            a = a + self.rng.normal(0, self.sigma, a.shape)
            self.sigma = max(0.1, self.sigma * 0.999)
        return np.clip(a, -1, 1), (X, A)

    def store(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))
        if len(self.buf) > 20000:
            self.buf.pop(0)

    def update(self, steps=1):
        if len(self.buf) < self.batch:
            return {}
        for _ in range(steps):
            idx = self.rng.choice(len(self.buf), self.batch, replace=False)
            items = [self.buf[i] for i in idx]
            Xs = torch.tensor(np.stack([it[0][0] for it in items]))
            As = torch.tensor(np.stack([it[0][1] for it in items]))
            a = torch.tensor(np.stack([it[1] for it in items]), dtype=torch.float32)
            r = torch.tensor([it[2] for it in items], dtype=torch.float32)
            X2 = torch.tensor(np.stack([it[3][0] for it in items]))
            A2 = torch.tensor(np.stack([it[3][1] for it in items]))
            d = torch.tensor([float(it[4]) for it in items])
            z = self.enc(Xs, As)
            with torch.no_grad():
                z2 = self.enc(X2, A2)
                y = r + self.gamma * (1 - d) * self.qt(z2, self.actor(z2))
            lq = ((self.q(z.detach(), a) - y) ** 2).mean()
            self.opt_q.zero_grad(); lq.backward(); self.opt_q.step()
            la = -self.q(z, self.actor(z)).mean()
            self.opt_a.zero_grad(); la.backward(); self.opt_a.step()
            with torch.no_grad():
                for pt, ps in zip(self.qt.parameters(), self.q.parameters()):
                    pt.mul_(0.995).add_(0.005 * ps)
        return {"lq": float(lq)}


# ---------------- non-graph A2C / PPO ----------------

class MLPPolicy(nn.Module):
    def __init__(self, sdim, adim, hid=128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(sdim, hid), nn.Tanh(),
                                  nn.Linear(hid, hid), nn.Tanh())
        self.mu = nn.Linear(hid, adim)
        self.logstd = nn.Parameter(torch.zeros(adim) - 0.5)
        self.v = nn.Linear(hid, 1)

    def forward(self, s):
        h = self.body(s)
        return self.mu(h), self.logstd.expand_as(self.mu(h)), self.v(h).squeeze(-1)


class NGAgent:
    """A2C or PPO on the flat normalized parameter vector."""
    def __init__(self, sdim, adim, kind="a2c", seed=0, lr=3e-4):
        torch.manual_seed(seed + 13)
        self.rng = np.random.default_rng(seed + 13)
        self.net = MLPPolicy(sdim, adim)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.kind = kind
        self.traj = []

    def act(self, s, explore=True):
        st = torch.tensor(s, dtype=torch.float32)
        with torch.no_grad():
            mu, logstd, v = self.net(st.unsqueeze(0))
            std = logstd.exp()
            u = mu + std * torch.randn_like(mu) if explore else mu
            a = torch.tanh(u)
            logp = (-0.5 * (((u - mu) / std) ** 2 + 2 * logstd
                    + np.log(2 * np.pi))
                    - torch.log(1 - a ** 2 + 1e-6)).sum(-1)
        return (a.detach().squeeze(0).numpy(), float(logp), float(v))

    def store(self, s, a, r, logp, v):
        self.traj.append((s, a, r, logp, v))

    def update(self):
        if len(self.traj) < 16:
            return {}
        S = torch.tensor(np.stack([t[0] for t in self.traj]), dtype=torch.float32)
        A = torch.tensor(np.stack([t[1] for t in self.traj]), dtype=torch.float32)
        R = torch.tensor([t[2] for t in self.traj], dtype=torch.float32)
        oldlp = torch.tensor([t[3] for t in self.traj], dtype=torch.float32)
        adv = R - R.mean()
        adv = adv / (adv.std() + 1e-6)
        for _ in range(4 if self.kind == "ppo" else 1):
            mu, logstd, v = self.net(S)
            std = logstd.exp()
            u = torch.atanh(torch.clamp(A, -0.999, 0.999))
            logp = (-0.5 * (((u - mu) / std) ** 2 + 2 * logstd + np.log(2 * np.pi))
                    - torch.log(1 - A ** 2 + 1e-6)).sum(-1)
            if self.kind == "ppo":
                ratio = torch.exp(logp - oldlp)
                l_pi = -torch.min(ratio * adv,
                                  torch.clamp(ratio, 0.8, 1.2) * adv).mean()
            else:
                l_pi = -(logp * adv).mean()
            loss = l_pi + 0.5 * ((v - R) ** 2).mean() - 0.001 * logstd.mean()
            self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.traj = []
        return {"loss": float(loss)}
