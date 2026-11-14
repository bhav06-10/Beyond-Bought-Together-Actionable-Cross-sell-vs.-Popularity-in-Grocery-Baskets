"""Feature builders shared by the M3 models. Every feature is computed from ONE fit window only
(January to June for tuning, January to September for the final test), so nothing from the
period being scored can leak in. Results are cached under data/processed/m3/."""
import pickle
import numpy as np, pandas as pd
from m3lib import OUT, SEED, T0, T1, T2, M3, cj_lines, items_index, between, pair_stats, bmatrix

WINDOWS = {"fit": (T0, T1), "fit2": (T0, T2)}        # fit2 = refit before the final test


def load():
    L = cj_lines(); return L, items_index(L)


def window_features(name, L=None, items=None):
    f = OUT / f"features_{name}.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    if L is None:
        L, items = load()
    start, end = WINDOWS[name]
    W = between(L, start, end)
    S = pair_stats(W, items, name=name)
    X, b = bmatrix(W, items)
    N = X.shape[0]
    pop = np.asarray(X.sum(0)).ravel() / N
    order = np.argsort(-pop)
    staples = {q: set(order[: max(1, int(round(q * len(items))))]) for q in M3["staple_grid"]}

    # item2vec equivalent: truncated SVD of the positive PMI matrix (Levy and Goldberg, 2014)
    O = S["O"]; prev = S["prev"]
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(O * N / np.outer(prev, prev))
    ppmi = np.where(np.isfinite(pmi) & (pmi > 0), pmi, 0); np.fill_diagonal(ppmi, 0)
    U, s, _ = np.linalg.svd(ppmi)
    emb = U[:, :32] * np.sqrt(s[:32])
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
    cos = emb @ emb.T

    # household histories: share of trips with each category, last purchase date, typical gap
    trips = W.drop_duplicates("basket").groupby("member").basket.size()
    members = pd.Index(sorted(trips.index))
    mi = members.get_indexer(W.member); ii = items.get_indexer(W.label)
    cnt = np.zeros((len(members), len(items))); np.add.at(cnt, (mi, ii), 1)
    hh_freq = cnt / trips.reindex(members).values[:, None]
    D = W.drop_duplicates(["member", "label", "date"]).sort_values("date")
    D = D.assign(gap=D.groupby(["member", "label"]).date.diff().dt.days)
    agg = D.groupby(["member", "label"]).agg(last=("date", "max"), gap_med=("gap", "median"), n=("date", "size"))
    last = np.full((len(members), len(items)), np.nan); gap = np.full_like(last, np.nan)
    r = members.get_indexer(agg.index.get_level_values(0)); c = items.get_indexer(agg.index.get_level_values(1))
    last[r, c] = (agg["last"] - T0).dt.days.values; gap[r, c] = agg.gap_med.values
    out = dict(name=name, start=start, end=end, items=items, N=N, pop=pop, staples=staples, S=S, emb=emb, cos=cos,
               members=members, trips=trips.reindex(members).values, hh_freq=hh_freq, last_day=last, gap_med=gap,
               dept=L.drop_duplicates("label").set_index("label").department.reindex(items).values)
    pickle.dump(out, open(f, "wb"))
    return out
