"""Step 3. The shared evaluation harness used by every RQ2 (and RQ4) model.

Protocol: every basket in the scored period with two or more categories becomes one query per
category: that category is hidden and the model ranks all categories not already in the rest of
the basket. Each hidden category is tagged
  repeat  - the household bought it in the fit window (a reminder could catch it)
  explore - new to the household (the cross-sell case this project is about)
  staple  - among the top 5% most common categories in the fit window (M2 definition)
Two protocols share these queries:
  open protocol       every query; the model ranks every category not already in the basket
  cross-sell protocol only explore, non-staple queries; every model ranks only categories that
                      are new to the household and not staples. This is the fair test of
                      cross-sell: plain popularity (B0) becomes the signal-free reference, so a
                      model only wins by finding real association, not by avoiding staples.
                      (The Kaggle placebo showed why: on non-staple targets, any model that merely
                      demotes staples beats an unrestricted popularity list.)
Metrics: Recall@10 (= hit rate, one hidden item per query), NDCG@10, mean popularity of the
list, catalogue coverage, and the share of list items from a department not in the basket.
"""
import numpy as np, pandas as pd
from m3lib import M3, bmatrix, between, rank_of, household_bootstrap

K = M3["k"]


def build_queries(L, items, start, end, F):
    W = between(L, start, end)
    sizes = W.groupby("basket").label.size()
    W = W[W.basket.isin(sizes.index[sizes >= 2])]
    X, b = bmatrix(W, items)
    meta = W.drop_duplicates("basket").set_index("basket").loc[b, ["member", "date", "store", "week"]].reset_index()
    rows = np.repeat(np.arange(X.shape[0]), np.diff(X.indptr))
    hidden = X.indices.copy()
    mrow = F["members"].get_indexer(meta.member.values)[rows]
    warm = mrow >= 0
    repeat = np.zeros(len(rows), bool)
    repeat[warm] = F["hh_freq"][mrow[warm], hidden[warm]] > 0
    st5 = np.isin(hidden, list(F["staples"][0.05]))
    return dict(X=X, baskets=b, meta=meta, row=rows, hidden=hidden, member=meta.member.values[rows], mrow=mrow,
                warm=warm, repeat=repeat, explore=warm & ~repeat, staple=st5, date=meta.date.values[rows],
                store=meta.store.values[rows], week=meta.week.values[rows], n=len(rows))


def context_mask(Q, idx):
    """Boolean (len(idx) x items): categories in the basket other than the hidden one."""
    M = Q["X"][Q["row"][idx]].toarray().astype(bool)
    M[np.arange(len(idx)), Q["hidden"][idx]] = False
    return M


def crosssell_queries(Q):
    return np.nonzero(Q["explore"] & ~Q["staple"])[0]


def evaluate(Q, score_fn, F, chunk=20000, crosssell=False):
    """Run a scorer over the queries. score_fn(Q, idx) -> (len(idx) x items) scores.
    crosssell=True scores only the cross-sell queries and hides staples and every category the
    household already bought from every model's list. Returns ranks and top-K lists, aligned
    with crosssell_queries(Q) when crosssell=True."""
    allq = crosssell_queries(Q) if crosssell else np.arange(Q["n"])
    ranks = np.empty(len(allq), int); top = np.empty((len(allq), K), int)
    pop = F["pop"]
    st = np.zeros(len(F["items"]), bool); st[list(F["staples"][0.05])] = True
    for s in range(0, len(allq), chunk):
        pos = np.arange(s, min(s + chunk, len(allq))); idx = allq[pos]
        S = np.asarray(score_fn(Q, idx), float) + 1e-9 * pop[None, :]     # popularity breaks ties
        S[context_mask(Q, idx)] = -np.inf
        if crosssell:
            S[:, st] = -np.inf
            m = Q["mrow"][idx]; w = m >= 0
            if w.any() and "hh_freq" in F:
                sub = S[w]; sub[F["hh_freq"][m[w]] > 0] = -np.inf; S[w] = sub
        ranks[pos] = rank_of(S, Q["hidden"][idx])
        top[pos] = np.argsort(-S, axis=1)[:, :K]
    return ranks, top


def query_metrics(Q, ranks):
    hit = (ranks <= K).astype(float)
    ndcg = np.where(ranks <= K, 1 / np.log2(ranks + 1), 0.0)
    return pd.DataFrame({"member": Q["member"], "hit": hit, "ndcg": ndcg, "warm": Q["warm"], "repeat": Q["repeat"],
                         "explore": Q["explore"], "staple": Q["staple"]})


SEGMENTS = {"all warm": lambda d: d.warm, "repeat": lambda d: d.repeat, "explore": lambda d: d.explore,
            "explore, non-staple": lambda d: d.explore & ~d.staple, "cold households": lambda d: ~d.warm}


def summarize(Q, ranks, top, F, name, ci=True, cs_ranks=None):
    d = query_metrics(Q, ranks)
    out = {"model": name}
    if cs_ranks is not None:
        qi = crosssell_queries(Q)
        c = pd.DataFrame({"member": Q["member"][qi], "hit": (cs_ranks <= K).astype(float),
                          "ndcg": np.where(cs_ranks <= K, 1 / np.log2(cs_ranks + 1), 0.0)})
        if ci:
            est, _ = household_bootstrap(c, ["hit", "ndcg"])
            out.update({"cross-sell | Recall@10": est.loc["hit", "mean"], "cross-sell | Recall@10 lo": est.loc["hit", "lo"],
                        "cross-sell | Recall@10 hi": est.loc["hit", "hi"], "cross-sell | NDCG@10": est.loc["ndcg", "mean"]})
        else:
            out.update({"cross-sell | Recall@10": c.hit.mean(), "cross-sell | NDCG@10": c.ndcg.mean()})
        out["cross-sell | queries"] = int(len(c))
    for seg, f in SEGMENTS.items():
        s = d[f(d)]
        if len(s) == 0:
            continue
        if ci:
            est, _ = household_bootstrap(s, ["hit", "ndcg"])
            out.update({f"{seg} | Recall@10": est.loc["hit", "mean"], f"{seg} | Recall@10 lo": est.loc["hit", "lo"],
                        f"{seg} | Recall@10 hi": est.loc["hit", "hi"], f"{seg} | NDCG@10": est.loc["ndcg", "mean"]})
        else:
            out.update({f"{seg} | Recall@10": s.hit.mean(), f"{seg} | NDCG@10": s.ndcg.mean()})
        out[f"{seg} | queries"] = int(len(s))
    out["list popularity"] = float(F["pop"][top].mean())
    out["coverage"] = float(len(np.unique(top)) / len(F["items"]))
    dept = F["dept"]
    ctx_dept = [set(dept[Q["X"][r].indices]) for r in range(Q["X"].shape[0])]
    samp = np.random.default_rng(0).choice(Q["n"], min(Q["n"], 30000), replace=False)
    out["cross-department share"] = float(np.mean([np.mean([dept[j] not in ctx_dept[Q["row"][q]] for j in top[q]]) for q in samp]))
    return out, d
