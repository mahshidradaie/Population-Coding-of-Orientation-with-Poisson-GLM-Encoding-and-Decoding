"""Population coding of orientation: Poisson-GLM encoding models and decoders.

Simulated V1-like population: N orientation-tuned neurons (von Mises tuning on the doubled angle),
Poisson spike counts in a 200 ms window, optional multiplicative shared gain noise (a source of noise
correlations). Encoding: per-neuron Poisson GLM (log link) on [cos 2th, sin 2th] fitted on training trials.
Decoding on held-out trials: maximum likelihood (grid search over fitted GLMs) and population vector.
Reference: Cramer-Rao bound from Fisher information of the true independent-Poisson population.
"""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import PoissonRegressor

WIN = 0.2
GRID = np.linspace(0, np.pi, 360, endpoint=False)

def true_rates(theta, N, kappa=1.5, rmax=40., rbase=3.):
    pref = np.linspace(0, np.pi, N, endpoint=False)
    return rbase + rmax * np.exp(kappa * (np.cos(2 * (np.asarray(theta)[..., None] - pref)) - 1))

def sample(theta, N, gain_sd, rng):
    lam = true_rates(theta, N) * WIN
    g = np.exp(gain_sd * rng.standard_normal((len(theta), 1)) - gain_sd ** 2 / 2)   # shared multiplicative gain
    return rng.poisson(lam * g)

def feats(theta): return np.stack([np.cos(2 * theta), np.sin(2 * theta)], -1)

def fit_glms(theta, counts):
    return [PoissonRegressor(alpha=1e-4, max_iter=300).fit(feats(theta), counts[:, i]) for i in range(counts.shape[1])]

def ml_decode(glms, counts):
    F = feats(GRID); loglam = np.stack([np.log(g.predict(F) + 1e-9) for g in glms], 1)   # (grid, N)
    lam = np.exp(loglam)
    ll = counts @ loglam.T - lam.sum(1)[None, :]                                       # Poisson log-lik (+const)
    return GRID[ll.argmax(1)]

def pop_vector(counts, N):
    pref = np.linspace(0, np.pi, N, endpoint=False)
    z = (counts * np.exp(2j * pref)).sum(1); return (np.angle(z) % (2 * np.pi)) / 2

def circ_err_deg(a, b):
    d = np.abs(a - b) % np.pi; d = np.minimum(d, np.pi - d); return np.degrees(d)

def crb_deg(N):
    th = GRID; eps = 1e-4
    df = (true_rates(th + eps, N) - true_rates(th - eps, N)) / (2 * eps)
    fi = (WIN * df ** 2 / true_rates(th, N)).sum(1).mean()
    return float(np.degrees(1 / np.sqrt(fi)))

if __name__ == "__main__":
    out = {}; Ns = (8, 16, 32, 64); rng = np.random.default_rng(0)
    for gain_sd in (0.0, 0.3):
        rows = {}
        for N in Ns:
            th_tr = rng.uniform(0, np.pi, 3000); th_te = rng.uniform(0, np.pi, 1000)
            Ctr, Cte = sample(th_tr, N, gain_sd, rng), sample(th_te, N, gain_sd, rng)
            glms = fit_glms(th_tr, Ctr)
            ml = np.sqrt(np.mean(circ_err_deg(ml_decode(glms, Cte), th_te) ** 2))
            pv = np.sqrt(np.mean(circ_err_deg(pop_vector(Cte, N), th_te) ** 2))
            rows[N] = dict(ml_rmse_deg=float(ml), popvec_rmse_deg=float(pv), cramer_rao_deg=crb_deg(N))
            print(gain_sd, N, rows[N], flush=True)
        out[f"shared_gain_sd_{gain_sd}"] = rows
    json.dump(out, open("results.json", "w"), indent=2)
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for key, ls in zip(out, ("-", "--")):
        ax.plot(Ns, [out[key][n]["ml_rmse_deg"] for n in Ns], "o" + ls, color="C0", label=f"GLM-ML ({key[-3:]})")
    ax.plot(Ns, [out[list(out)[0]][n]["cramer_rao_deg"] for n in Ns], "k:", label="Cramer-Rao (indep.)")
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("neurons N"); ax.set_ylabel("decoding RMSE (deg)")
    ax.legend(fontsize=7); plt.tight_layout(); plt.savefig("decoding_error.png", dpi=150)
