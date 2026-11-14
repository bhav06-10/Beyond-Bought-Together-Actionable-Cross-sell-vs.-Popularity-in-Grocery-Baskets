"""Shared Milestone 3 helpers: periods, basket matrices on a fixed item index, pair statistics
(swap-randomized chance baseline, M2 shrunk strength, DuMouchel's EBGM), ranking metrics and
the paired household bootstrap. Every M3 script imports from here."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # src/ for common.py and lib.py
import numpy as np, pandas as pd
from scipy import sparse, stats, optimize, special
from common import CFG, P, ROOT, save_numbers, BLUE, ORANGE, GREEN, GREY, INK, MUTED, plt   # noqa: F401
from lib import swap_null, fit_alpha

M3 = CFG["m3"]; PER = M3["periods"]; SEED = CFG["seed"]
OUT = P["processed"] / "m3"; OUT.mkdir(parents=True, exist_ok=True)
TAB = P["tables"]; FIG = P["figures"]
T0, T1, T2, T3 = (pd.Timestamp(PER[k]) for k in ["fit_start", "fit_end", "tune_end", "test_end"])


def m3_numbers(key, d):
    """Headline numbers go to reports/numbers.json under m3_<key>."""
    save_numbers(f"m3_{key}", d)


# ---------------------------------------------------------------- data access
def cj_lines():
    """Category-level CJ lines from M2 (one row per basket and category) with store and week added."""
    L = pd.read_csv(P["processed"] / "cj_lines.csv.gz", dtype={"basket": str, "member": str},
                    parse_dates=["date", "transaction_timestamp"])
    B = pd.read_csv(P["processed"] / "cj_baskets.csv.gz", dtype={"basket": str, "member": str, "store": str})
    return L.merge(B[["basket", "store", "week"]], on="basket", how="left")


def items_index(L):
    return pd.Index(sorted(L.label.unique()))


def between(L, start, end, col="date"):
    return L[(L[col] >= start) & (L[col] < end)]


def bmatrix(L, items, baskets=None):
    """Binary basket x item matrix on a FIXED item index, so periods line up."""
    b = pd.Index(L.basket.unique()) if baskets is None else baskets
    X = sparse.csr_matrix((np.ones(len(L), np.float32), (b.get_indexer(L.basket), items.get_indexer(L.label))),
                          shape=(len(b), len(items)))
    X.data[:] = 1
    return X, b


# ---------------------------------------------------------------- pair statistics
def ebgm_fit(n, e):
    """DuMouchel (1999) Gamma-Poisson Shrinker: n ~ Poisson(lambda*e), lambda ~ mixture of two gammas.
    Fit the five prior parameters by maximum likelihood of the negative-binomial mixture."""
    n = np.asarray(n, float); e = np.maximum(np.asarray(e, float), 1e-6)

    def nll(th):
        a1, b1, a2, b2 = np.exp(th[:4]); w = special.expit(th[4])
        l1 = stats.nbinom.logpmf(n, a1, b1 / (b1 + e)); l2 = stats.nbinom.logpmf(n, a2, b2 / (b2 + e))
        return -np.logaddexp(np.log(w) + l1, np.log1p(-w) + l2).sum()
    best = None
    for start in ([np.log(0.2), np.log(0.1), np.log(2), np.log(4), 1.5], [np.log(1), np.log(1), np.log(5), np.log(5), 0.0]):
        r = optimize.minimize(nll, start, method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-6, fatol=1e-6))
        if best is None or r.fun < best.fun:
            best = r
    a1, b1, a2, b2 = np.exp(best.x[:4]); w = special.expit(best.x[4])
    return dict(a1=a1, b1=b1, a2=a2, b2=b2, w=w)


def ebgm(n, e, th):
    """Posterior geometric mean of lambda (EBGM) and the 5th percentile (EB05) given the fitted prior."""
    n = np.asarray(n, float); e = np.maximum(np.asarray(e, float), 1e-6)
    l1 = stats.nbinom.logpmf(n, th["a1"], th["b1"] / (th["b1"] + e)) + np.log(th["w"])
    l2 = stats.nbinom.logpmf(n, th["a2"], th["b2"] / (th["b2"] + e)) + np.log1p(-th["w"])
    q = np.exp(l1 - np.logaddexp(l1, l2))
    elog = q * (special.digamma(th["a1"] + n) - np.log(th["b1"] + e)) + (1 - q) * (special.digamma(th["a2"] + n) - np.log(th["b2"] + e))
    return np.exp(elog)


def pair_stats(L, items, samples=None, seed=SEED, name=None):
    """Observed pair counts, swap-randomized expectation, M2 strength, EBGM, lift and confidence as
    dense item x item matrices. Cached in data/processed/m3/pairs_<name>.npz."""
    f = OUT / f"pairs_{name}.npz" if name else None
    if f is not None and f.exists():
        z = np.load(f, allow_pickle=True); return {k: z[k] for k in z.files}
    X, _ = bmatrix(L, items)
    N = X.shape[0]
    O = (X.T @ X).toarray().astype(float)
    E, SD = swap_null(X, samples or M3["swap_samples"], seed)
    prev = np.diag(O).copy()
    off = ~np.eye(len(items), dtype=bool)
    iu = np.triu_indices(len(items), 1)
    alpha = fit_alpha(O[iu], np.maximum(E[iu], 1e-6))
    strength = (O + alpha) / (E + alpha)
    th = ebgm_fit(O[iu], E[iu])
    EB = np.ones_like(O); EB[iu] = ebgm(O[iu], E[iu], th); EB = np.triu(EB, 1) + np.triu(EB, 1).T
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = np.where(np.outer(prev, prev) > 0, O * N / np.outer(prev, prev), 1.0)
        conf = np.where(prev[:, None] > 0, O / prev[:, None], 0.0)        # conf[a, b] = P(b | a)
        p = stats.poisson.sf(O - 1, np.maximum(E, 1e-6))
    for M in (strength, EB, lift, conf):
        M[~off] = 0 if M is conf else 1
    res = dict(O=O, E=E, SD=SD, prev=prev, N=np.array(N), strength=strength, ebgm=EB, lift=lift, conf=conf, p=p,
               alpha=np.array(alpha), ebgm_prior=np.array([th[k] for k in ["a1", "b1", "a2", "b2", "w"]]))
    if f is not None:
        np.savez_compressed(f, **res)
    return res


def pair_frame(S, items, min_n=1):
    """Upper-triangle table of a pair_stats result, with FDR-adjusted significance."""
    iu = np.triu_indices(len(items), 1)
    t = pd.DataFrame({"a": items[iu[0]], "b": items[iu[1]], "n": S["O"][iu], "expected": S["E"][iu],
                      "prev_a": S["prev"][iu[0]], "prev_b": S["prev"][iu[1]], "lift": S["lift"][iu],
                      "strength": S["strength"][iu], "ebgm": S["ebgm"][iu], "p": S["p"][iu]})
    t["ratio"] = t.n / np.maximum(t.expected, 1e-6)
    t = t[t.n >= min_n].copy()
    t["q"] = stats.false_discovery_control(t.p.clip(0, 1).values)
    t["sig_pos"] = (t.q < CFG["rules"]["fdr"]) & (t.ratio > 1)
    return t.reset_index(drop=True)


# ---------------------------------------------------------------- ranking evaluation
def rank_of(scores, truth):
    """1-based rank of the true column in each row of a score matrix (ties broken pessimistically)."""
    s_true = scores[np.arange(len(truth)), truth]
    return (scores > s_true[:, None]).sum(1) + (scores == s_true[:, None]).sum(1)


def household_bootstrap(df, value_cols, group="member", reps=None, seed=SEED):
    """Mean of each column with a 95% interval from resampling households (paired across columns)."""
    reps = reps or M3["bootstrap"]
    g = df.groupby(group)[value_cols].agg(["sum", "count"])
    sums = np.stack([g[(c, "sum")].values for c in value_cols], 1); cnt = g[(value_cols[0], "count")].values
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(cnt), size=(reps, len(cnt)))
    boots = np.stack([sums[i].sum(0) / cnt[i].sum() for i in idx])
    est = sums.sum(0) / cnt.sum()
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return pd.DataFrame({"mean": est, "lo": lo, "hi": hi}, index=value_cols), boots


def holm(pvals):
    p = np.asarray(pvals); order = np.argsort(p); adj = np.empty_like(p, dtype=float)
    run = 0
    for r, i in enumerate(order):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(1, run)
    return adj
