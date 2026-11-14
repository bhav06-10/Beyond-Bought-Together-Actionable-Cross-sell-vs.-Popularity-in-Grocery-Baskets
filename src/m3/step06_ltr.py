"""Step 6. C3: a LightGBM LambdaMART learning-to-rank model over the candidates proposed by
B0, B1, C1 and C2. Trained on tuning-quarter queries with features from January to June only.

Tuning: random search (trials in config.yaml) over leaves, learning rate, minimum child samples
and feature fraction; early stopping on NDCG@10 of a 20% household-held-out slice of the tuning
queries; the trial is chosen with the same selection score as C1 and C2.
"""
import json, pickle
import numpy as np, pandas as pd, lightgbm as lgb
from m3lib import OUT, TAB, M3, SEED, m3_numbers
from features import load, window_features
from rq2lib import b0, b1, c1, c2, c3_features, Promo, FEATURES

cfg = M3["c3"]; K = M3["k"]
L, items = load(); F = window_features("fit", L, items)
Q = pickle.load(open(OUT / "queries_tune.pkl", "rb"))
best = json.load(open(OUT / "best_params.json"))
p1 = best["C1"]; p2 = best["C2"]
scorers = {"b0": b0(F), "b1": b1(F), "c1": c1(F, "fit", p1["min_support"], p1["min_confidence"]),
           "c2": c2(F, p2["measure"], p2["prior_weight"], p2["clip"])}
promo = Promo()
rng = np.random.default_rng(SEED)
sample = np.sort(rng.choice(Q["n"], cfg["train_queries"], replace=False))


def build(idx):
    parts = []
    for s in range(0, len(idx), 2000):
        qi, cj, X, y, _ = c3_features(Q, idx[s:s + 2000], F, scorers, promo, cfg["candidates"])
        parts.append((idx[s:s + 2000][qi], cj, X, y))
    q = np.concatenate([p[0] for p in parts]); return q, np.concatenate([p[1] for p in parts]), np.vstack([p[2] for p in parts]), np.concatenate([p[3] for p in parts])


q, cj, X, y = build(sample)
print("rows", len(y), "queries", len(sample), "truth in candidates", y.sum() / len(sample), flush=True)
members = np.unique(Q["member"][sample])
val_m = set(rng.choice(members, len(members) // 5, replace=False))
is_val = np.array([m in val_m for m in Q["member"][q]])


def groups(qq):
    _, c = np.unique(qq, return_counts=True); return c        # q is sorted, so groups are contiguous


def recall_by_query(qq, yy, score, queries, keep=None):
    """Rank of the hidden category among this query's candidates; absent from candidates = miss.
    keep: optional row filter (the cross-sell protocol keeps only new, non-staple candidates)."""
    d = pd.DataFrame({"q": qq, "y": yy, "s": score})
    if keep is not None:
        d = d[keep]
    d["r"] = d.groupby("q").s.rank(ascending=False, method="max")
    hit = d[d.y == 1].set_index("q").r.le(K).astype(float)
    return pd.Series(0.0, index=queries).add(hit, fill_value=0).reindex(queries).values


def selection(hits, cs_hits, queries):
    w = Q["warm"][queries]; e = Q["explore"][queries] & ~Q["staple"][queries]
    return 0.5 * (hits[w].mean() + cs_hits[e].mean()), hits[w].mean(), cs_hits[e].mean()


tr, va = ~is_val, is_val
vq = np.unique(q[va])
space = dict(num_leaves=[15, 31, 63], learning_rate=[0.05, 0.1], min_child_samples=[20, 50, 100],
             colsample_bytree=[0.7, 0.9, 1.0])
trials = []
for t in range(cfg["trials"]):
    par = {k: v[rng.integers(len(v))] for k, v in space.items()}
    m = lgb.LGBMRanker(objective="lambdarank", n_estimators=400, random_state=SEED, verbose=-1, n_jobs=2, **par)
    m.fit(X[tr], y[tr], group=groups(q[tr]), eval_set=[(X[va], y[va])], eval_group=[groups(q[va])],
          eval_at=[K], callbacks=[lgb.early_stopping(30, verbose=False)])
    sc = m.predict(X[va], num_iteration=m.best_iteration_)
    hits = recall_by_query(q[va], y[va], sc, vq)
    new_ns = (X[va][:, FEATURES.index("repeat")] == 0) & (X[va][:, FEATURES.index("staple")] == 0)
    cs_hits = recall_by_query(q[va], y[va], sc, vq, keep=new_ns)
    s, w, e = selection(hits, cs_hits, vq)
    trials.append(dict(trial=t, **par, best_iteration=int(m.best_iteration_), selection=s, warm_recall=w, crosssell_recall=e))
    print(trials[-1], flush=True)
T = pd.DataFrame(trials).sort_values("selection", ascending=False); T.to_csv(TAB / "m3_rq2_tuning_c3.csv", index=False)
b = T.iloc[0]
par = {k: (int(b[k]) if k in ("num_leaves", "min_child_samples") else float(b[k])) for k in space}
final = lgb.LGBMRanker(objective="lambdarank", n_estimators=int(b.best_iteration), random_state=SEED, verbose=-1, n_jobs=2, **par)
final.fit(X, y, group=groups(q))
final.booster_.save_model(str(OUT / "c3_model.txt"))
best["C3"] = dict(**par, n_estimators=int(b.best_iteration), selection=float(b.selection))
json.dump(best, open(OUT / "best_params.json", "w"), indent=2)
imp = pd.DataFrame({"feature": FEATURES, "gain": final.booster_.feature_importance("gain")}).sort_values("gain", ascending=False)
imp.to_csv(TAB / "m3_rq2_c3_importance.csv", index=False)
m3_numbers("tuning_c3", dict(best=best["C3"], train_rows=int(len(y)), train_queries=int(len(sample)),
                             truth_in_candidates=float(y.sum() / len(sample))))
print(best["C3"]); print(imp.head(10))
