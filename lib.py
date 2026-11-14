"""Baskets, the swap-randomization chance baseline, and pair measures (steps 4, 7, 8, 9)."""
import numpy as np, pandas as pd
from scipy import sparse, stats
from common import P


def load_lines(source):
    dates = {"kaggle": ["date"], "arules": [], "cj": ["date", "transaction_timestamp"]}[source]
    return pd.read_csv(P["processed"] / f"{source}_lines.csv.gz", dtype={"basket": str, "member": str},
                       parse_dates=dates)


def basket_matrix(lines):
    """Binary basket x item matrix (CSR), plus the basket and item indexes."""
    b = pd.Index(lines.basket.unique()); items = pd.Index(sorted(lines.label.unique()))
    X = sparse.csr_matrix((np.ones(len(lines), np.float32), (b.get_indexer(lines.basket), items.get_indexer(lines.label))),
                          shape=(len(b), len(items)))
    X.data[:] = 1
    return X, b, items


def swap_null(X, samples=40, seed=42):
    """Swap randomization (curveball algorithm): random basket data that keeps every basket's
    size and every item's basket count exactly, so only the pairing of items is random.
    Returns the mean and standard deviation of each pair count over the samples."""
    rng = np.random.default_rng(seed)
    rows = [set(X.indices[X.indptr[i]:X.indptr[i + 1]]) for i in range(X.shape[0])]
    B, I = X.shape

    def trade(k):
        pick = rng.integers(B, size=(k, 2))
        for i, j in pick:
            if i == j:
                continue
            a, b = rows[i], rows[j]
            only_a, only_b = a - b, b - a
            if not only_a or not only_b:
                continue
            pool = list(only_a | only_b); rng.shuffle(pool)
            both = a & b; n = len(only_a)
            rows[i] = both | set(pool[:n]); rows[j] = both | set(pool[n:])

    trade(5 * B)                                        # burn-in
    s1 = np.zeros((I, I)); s2 = np.zeros((I, I))
    for _ in range(samples):
        trade(B // 2)                                   # thinning between samples
        r = np.repeat(np.arange(B), [len(x) for x in rows])
        c = np.fromiter((k for x in rows for k in x), int)
        M = sparse.csr_matrix((np.ones(len(c), np.float32), (r, c)), shape=(B, I))
        C = (M.T @ M).toarray()
        s1 += C; s2 += C * C
    mean = s1 / samples
    return mean, np.sqrt(np.maximum(s2 / samples - mean ** 2, 0))


def pair_table(X, items, min_baskets=1, samples=40):
    """Observed co-occurrence, chance expectation, z, p, shrunk strength, lift, Kulczynski."""
    O = (X.T @ X).toarray()
    E, SD = swap_null(X, samples)
    prev = np.diag(O).copy()
    iu = np.triu_indices(len(items), 1)
    t = pd.DataFrame({"a": items[iu[0]], "b": items[iu[1]], "n": O[iu], "expected": E[iu],
                      "sd": SD[iu], "prev_a": prev[iu[0]], "prev_b": prev[iu[1]]})
    t = t[t.n >= min_baskets].copy()
    N = X.shape[0]
    t["z"] = (t.n - t.expected) / np.maximum(t.sd, 0.5)
    t["p"] = stats.poisson.sf(t.n - 1, np.maximum(t.expected, 1e-6))   # P(count >= n by chance)
    t["lift"] = t.n * N / (t.prev_a * t.prev_b)               # classic lift, for comparison
    t["ratio"] = t.n / np.maximum(t.expected, 1e-6)            # observed / chance
    t["kulc"] = 0.5 * (t.n / t.prev_a + t.n / t.prev_b)       # null-invariant measure from M1
    t["ir"] = (t.prev_a - t.prev_b).abs() / (t.prev_a + t.prev_b - t.n)   # imbalance ratio
    # empirical-Bayes shrinkage: n ~ Poisson(lambda * E), lambda ~ Gamma(alpha, alpha)
    alpha = fit_alpha(t.n.values, np.maximum(t.expected.values, 1e-6))
    t["strength"] = (t.n + alpha) / (t.expected + alpha)
    t.attrs["alpha"] = alpha
    t["q"] = stats.false_discovery_control(t.p.values)
    return t.reset_index(drop=True)


def fit_alpha(n, e):
    """Pick the Gamma prior strength that maximises the negative-binomial likelihood."""
    grid = np.logspace(-1, 3, 81)
    ll = [stats.nbinom.logpmf(n, a, a / (a + e)).sum() for a in grid]
    return float(grid[int(np.argmax(ll))])


def prevalence(X, items):
    return pd.Series(np.asarray(X.sum(0)).ravel() / X.shape[0], index=items).sort_values(ascending=False)
