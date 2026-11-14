"""RQ2 models (task type: ranking / basket completion). Each model is a scorer that, for a batch
of queries, returns a score for every category; the harness hides the basket's own items.

  B0  global popularity                         (baseline: "what everyone buys")
  B1  household's own purchase frequency        (baseline: a pure reminder engine)
  C1  association rules from FP-Growth          (candidate: industry "frequently bought together")
  C2  null-calibrated naive Bayes               (candidate: the M2/EBGM measure turned into a recommender)
  C3  LightGBM LambdaMART hybrid                (candidate: learns when to remind and when to cross-sell)
"""
from functools import lru_cache
import numpy as np, pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from m3lib import OUT, T0, M3, between, bmatrix
from harness import context_mask

# ---------------------------------------------------------------- B0, B1
def b0(F):
    return lambda Q, idx: np.tile(F["pop"], (len(idx), 1))


def b1(F):
    def f(Q, idx):
        S = np.tile(F["pop"] * 1e-3, (len(idx), 1))          # cold households fall back to popularity
        m = Q["mrow"][idx]; w = m >= 0
        S[w] += F["hh_freq"][m[w]]
        return S
    return f


# ---------------------------------------------------------------- helpers: aggregate over the context
def ctx_sum(Q, idx, M):
    """sum over context items a of M[a, :]  (context = basket minus hidden)."""
    Xr = Q["X"][Q["row"][idx]]
    return np.asarray(Xr @ M) - M[Q["hidden"][idx]]


def ctx_max(Q, idx, M):
    """max over context items a of M[a, :], via the top-two trick per basket."""
    out = np.zeros((len(idx), M.shape[1]))
    rows = Q["row"][idx]; hid = Q["hidden"][idx]
    X = Q["X"]
    cache = {}
    for k, (r, h) in enumerate(zip(rows, hid)):
        if r not in cache:
            its = X.indices[X.indptr[r]:X.indptr[r + 1]]
            R = M[its]
            o = np.argsort(-R, axis=0)
            cache[r] = (its, R[o[0], np.arange(R.shape[1])], R[o[1], np.arange(R.shape[1])] if len(its) > 1 else np.zeros(R.shape[1]), its[o[0]])
        its, m1, m2, arg = cache[r]
        out[k] = np.where(arg == h, m2, m1)
    return out


# ---------------------------------------------------------------- C1: FP-Growth association rules
@lru_cache(maxsize=8)
def _rules(window, min_support):
    from features import window_features, load
    f = OUT / f"c1_rules_{window}_{min_support}.pkl"
    if f.exists():
        return pd.read_pickle(f)
    L, items = load(); F = window_features(window, L, items)
    W = between(L, F["start"], F["end"])
    X, _ = bmatrix(W, items)
    df = pd.DataFrame.sparse.from_spmatrix(X.astype(bool), columns=list(range(len(items)))).sparse.to_dense().astype(bool)
    fi = fpgrowth(df, min_support=min_support, max_len=3, use_colnames=True)
    r = association_rules(fi, metric="lift", min_threshold=1.0, num_itemsets=len(df))
    r = r[r.consequents.apply(len) == 1].copy()
    r["b"] = r.consequents.apply(lambda s: next(iter(s)))
    r["ante"] = r.antecedents.apply(lambda s: tuple(sorted(s)))
    r = r[["ante", "b", "support", "confidence", "lift"]].reset_index(drop=True)
    r.to_pickle(f)
    return r


def c1(F, window, min_support, min_confidence):
    r = _rules(window, min_support)
    r = r[r.confidence >= min_confidence]
    I = len(F["items"])
    C = np.zeros((I, I))
    one = r[r.ante.apply(len) == 1]
    C[[a[0] for a in one.ante], one.b.values] = one.confidence.values
    two = {}
    for a, b, c in zip(r.ante, r.b, r.confidence):
        if len(a) == 2:
            two.setdefault(a, np.zeros(I))[b] = max(two.get(a, np.zeros(I))[b], c)

    def f(Q, idx):
        S = ctx_max(Q, idx, C)
        if not two:
            return S
        X = Q["X"]; cache = {}
        for k, (rw, h) in enumerate(zip(Q["row"][idx], Q["hidden"][idx])):
            if rw not in cache:                        # two-item antecedents present in this basket
                its = sorted(X.indices[X.indptr[rw]:X.indptr[rw + 1]])
                cache[rw] = [(a, two[a]) for a in ((x, y) for i, x in enumerate(its) for y in its[i + 1:]) if a in two]
            for a, v in cache[rw]:
                if h not in a:
                    S[k] = np.maximum(S[k], v)
        return S
    return f


# ---------------------------------------------------------------- C2: null-calibrated strength
def c2_matrix(F, measure, clip=True):
    """log of the shrunk observed/chance ratio. clip=True keeps only above-chance evidence."""
    M = F["S"][measure]
    W = np.log(np.maximum(M, 1.0) if clip else np.maximum(M, 1e-3))
    np.fill_diagonal(W, 0)
    return W


def c2(F, measure="ebgm", prior_weight=1.0, clip=True):
    """Null-calibrated naive Bayes: log P(b) * prior_weight + sum over basket items a of
    log(shrunk ratio(a, b)). With prior_weight 0 it ranks by association alone; with 1 it is the
    naive-Bayes estimate of P(b | basket) with every pairwise ratio replaced by its empirical-Bayes
    shrunk version measured against the swap-randomized baseline."""
    W = c2_matrix(F, measure, clip)
    lp = np.log(np.maximum(F["pop"], 1e-6))

    def f(Q, idx):
        return ctx_sum(Q, idx, W) + prior_weight * lp[None, :]
    return f


# ---------------------------------------------------------------- C3 features
class Promo:
    def __init__(self):
        pc = pd.read_csv(OUT / "cj_promo_cat.csv.gz", dtype={"store": str})
        from features import load
        _, items = load()
        self.stores = pd.Index(sorted(pc.store.unique()))
        self.A = np.full((len(self.stores), 54, len(items), 2), np.nan, np.float32)
        s = self.stores.get_indexer(pc.store); i = items.get_indexer(pc.label); ok = i >= 0
        self.A[s[ok], pc.week.values[ok], i[ok], 0] = pc.disp_share.values[ok]
        self.A[s[ok], pc.week.values[ok], i[ok], 1] = pc.mail_share.values[ok]

    def get(self, store, week):
        s = self.stores.get_indexer(store); out = np.full((len(store), self.A.shape[2], 2), np.nan, np.float32)
        ok = s >= 0
        out[ok] = self.A[s[ok], week[ok]]
        return out


FEATURES = ["pop", "hh_freq", "repeat", "c1", "c2_sum", "c2_max", "ebgm_sum", "cos_mean", "cos_max", "days_since",
            "gap_med", "clock", "staple", "same_dept", "ctx_size", "disp_share", "mail_share", "hh_trips"]


def c3_features(Q, idx, F, scorers, promo, n_cand):
    """Candidate generation (union of the top lists of B0, B1, C1, C2) and the feature matrix."""
    I = len(F["items"])
    ctx = context_mask(Q, idx)
    S = {k: np.asarray(fn(Q, idx), float) for k, fn in scorers.items()}
    S["ebgm_sum"] = ctx_sum(Q, idx, c2_matrix(F, "ebgm"))
    S["c2_max"] = ctx_max(Q, idx, c2_matrix(F, "ebgm"))
    cosm = F["cos"].copy(); np.fill_diagonal(cosm, 0)
    S["cos_sum"] = ctx_sum(Q, idx, cosm); S["cos_max"] = ctx_max(Q, idx, cosm)
    pre = np.zeros((len(idx), I))
    m_all = Q["mrow"][idx]; bought = np.zeros((len(idx), I), bool)
    bought[m_all >= 0] = F["hh_freq"][m_all[m_all >= 0]] > 0
    st = np.zeros(I, bool); st[list(F["staples"][0.05])] = True
    newns = ~bought & ~st[None, :]                       # cross-sell pool: new to the household, not a staple
    for k, w, pool in [("b0", n_cand // 5, None), ("b1", n_cand * 2 // 5, None), ("c1", n_cand // 5, None), ("c2", n_cand // 5, None),
                       ("c1", n_cand // 5, newns), ("c2", n_cand // 5, newns)]:
        s = np.where(ctx, -np.inf, S[k] + 1e-9 * F["pop"])
        if pool is not None:
            s = np.where(pool, s, -np.inf)
        top = np.argsort(-s, 1)[:, :w]
        val = np.maximum(np.take_along_axis(pre, top, 1), np.isfinite(np.take_along_axis(s, top, 1)).astype(float))
        np.put_along_axis(pre, top, val, 1)
    pre[ctx] = 0
    qi, cj = np.nonzero(pre)
    m = Q["mrow"][idx][qi]; warm = m >= 0
    n_ctx = ctx.sum(1)
    day = ((pd.to_datetime(Q["date"][idx]) - T0).days.values)[qi]
    hf = np.zeros(len(qi)); hf[warm] = F["hh_freq"][m[warm], cj[warm]]
    last = np.full(len(qi), np.nan); last[warm] = F["last_day"][m[warm], cj[warm]]
    gap = np.full(len(qi), np.nan); gap[warm] = F["gap_med"][m[warm], cj[warm]]
    since = day - last
    dept = F["dept"]
    same = np.array([np.mean(dept[np.nonzero(ctx[q])[0]] == dept[j]) for q, j in zip(qi, cj)])
    pr = promo.get(Q["store"][idx][qi], Q["week"][idx][qi].astype(int))
    pr = pr[np.arange(len(qi)), cj]
    trips = np.zeros(len(qi)); trips[warm] = F["trips"][m[warm]]
    Xf = np.column_stack([F["pop"][cj], hf, (hf > 0).astype(float), S["c1"][qi, cj], S["c2"][qi, cj], S["c2_max"][qi, cj],
                          S["ebgm_sum"][qi, cj], S["cos_sum"][qi, cj] / np.maximum(n_ctx[qi], 1), S["cos_max"][qi, cj], since, gap,
                          since / gap, np.isin(cj, list(F["staples"][0.05])).astype(float), same, n_ctx[qi], pr[:, 0], pr[:, 1], trips])
    y = (cj == Q["hidden"][idx][qi]).astype(int)
    return qi, cj, Xf, y, S["c2"]
