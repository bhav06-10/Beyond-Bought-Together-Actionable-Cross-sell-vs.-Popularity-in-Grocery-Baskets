"""Step 13. RQ4: does the order of trips add anything beyond household taste?
Task types: ranking (next-trip prediction) and clustering (shopping missions).

Query     every trip in the scored quarter; the model sees the household's history and the
          previous trip, and ranks categories for the next trip
Baseline  taste only: the household's own purchase frequency (fit window)
Candidates  first-order Markov model of trip-to-trip transitions (sequence only);
          NMF shopping missions (basket-level clusters; a household = its mix of missions);
          taste + Markov (does sequence add to taste?)
Metric    Recall@10 over the categories in the next trip, overall and for categories new to the
          household; paired household bootstrap for "taste + Markov minus taste"
Clusters  mission count k chosen on the tuning quarter; stability = matched cosine of mission
          profiles and adjusted Rand index of basket labels across 5 bootstrap refits
"""
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF
from sklearn.metrics import adjusted_rand_score
from m3lib import (TAB, FIG, SEED, M3, T1, T2, T3, m3_numbers, between, bmatrix, household_bootstrap, plt, BLUE, ORANGE, GREEN, GREY, MUTED)
from features import load, window_features

K = M3["k"]; rng = np.random.default_rng(SEED)
L, items = load()
trips = L.drop_duplicates("basket")[["basket", "member", "transaction_timestamp"]].sort_values(["member", "transaction_timestamp"])
trips["prev"] = trips.groupby("member").basket.shift()
Xall, ball = bmatrix(L, items)
row = pd.Series(np.arange(len(ball)), index=ball)


def markov(F):
    W = trips[(trips.transaction_timestamp >= F["start"]) & (trips.transaction_timestamp < F["end"])].dropna(subset=["prev"])
    A = Xall[row[W.prev].values]; B = Xall[row[W.basket].values]
    C = (A.T @ B).toarray(); na = np.asarray(A.sum(0)).ravel()
    return C / np.maximum(na[:, None], 1)                          # P(b in next trip | a in this trip)


def missions(F, k, sample=None, seed=SEED):
    W = between(L, F["start"], F["end"]); X, b = bmatrix(W, items)
    if sample is not None:
        X = X[sample]
    Xn = X.multiply(1 / np.maximum(X.sum(1), 1)).tocsr()
    m = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=300).fit(Xn)
    return m, Xn, b


def queries(s, e, F):
    Qt = trips[(trips.transaction_timestamp >= s) & (trips.transaction_timestamp < e)].dropna(subset=["prev"])
    Y = Xall[row[Qt.basket].values].toarray().astype(bool); P = Xall[row[Qt.prev].values]
    mrow = F["members"].get_indexer(Qt.member)
    seen = np.zeros_like(Y); w = mrow >= 0; seen[w] = F["hh_freq"][mrow[w]] > 0
    return dict(Y=Y, P=P, mrow=mrow, member=Qt.member.values, seen=seen, baskets=Qt.basket.values, prevb=Qt.prev.values)


def taste(F, Q):
    S = np.tile(F["pop"] * 1e-3, (len(Q["mrow"]), 1)); w = Q["mrow"] >= 0; S[w] += F["hh_freq"][Q["mrow"][w]]
    return S


def seq(Mk, Q):
    n = np.maximum(np.asarray(Q["P"].sum(1)).ravel(), 1)
    return np.asarray(Q["P"] @ Mk) / n[:, None]


def mission_scores(F, Q, model):
    """household mission mix (mean over its fit-window baskets) times mission profiles."""
    W = between(L, F["start"], F["end"]); X, b = bmatrix(W, items)
    Wm = model.transform(X.multiply(1 / np.maximum(X.sum(1), 1)).tocsr())
    mem = W.drop_duplicates("basket").set_index("basket").loc[b, "member"].values
    mix = pd.DataFrame(Wm).groupby(mem).mean()
    M = mix.reindex(Q["member"]).fillna(mix.mean()).values
    return M @ model.components_


def score(Q, S):
    top = np.argsort(-S, 1)[:, :K]
    hit = np.take_along_axis(Q["Y"], top, 1)
    nY = Q["Y"].sum(1)
    rec = hit.sum(1) / np.minimum(nY, K)
    ex = Q["Y"] & ~Q["seen"]; nE = ex.sum(1)
    hitE = np.take_along_axis(ex, top, 1).sum(1)
    return pd.DataFrame({"member": Q["member"], "recall": rec, "explore_recall": np.where(nE > 0, hitE / np.maximum(nE, 1), np.nan),
                         "has_explore": nE > 0, "warm": Q["mrow"] >= 0})


def norm(S):
    return S / np.maximum(S.max(1, keepdims=True), 1e-12)


# ---- tuning on July to September (fit window January to June) ------------------------------
F = window_features("fit", L, items); Qv = queries(T1, T2, F); Mk = markov(F)
base_v = taste(F, Qv); seq_v = seq(Mk, Qv)
tune = []
for lam in [0.25, 0.5, 1.0, 2.0]:
    d = score(Qv, norm(base_v) + lam * norm(seq_v)); tune.append(dict(model="taste+markov", param=lam, recall=d.recall.mean(), explore=d.explore_recall.mean()))
nmf_models = {}
for k in M3["rq4"]["nmf_k"]:
    m, _, _ = missions(F, k); nmf_models[k] = m
    d = score(Qv, mission_scores(F, Qv, m)); tune.append(dict(model="nmf", param=k, recall=d.recall.mean(), explore=d.explore_recall.mean()))
TU = pd.DataFrame(tune); TU["selection"] = 0.5 * (TU.recall + TU.explore); TU.to_csv(TAB / "m3_rq4_tuning.csv", index=False); print(TU)
lam = TU[TU.model == "taste+markov"].sort_values("selection").param.iloc[-1]
kbest = int(TU[TU.model == "nmf"].sort_values("selection").param.iloc[-1])

# ---- mission stability for each k (bootstrap refits on fit-window baskets) ---------------------
stab = []
for k in M3["rq4"]["nmf_k"]:
    ref, Xn, _ = missions(F, k); lab0 = ref.transform(Xn).argmax(1)
    for r in range(5):
        smp = rng.choice(Xn.shape[0], Xn.shape[0], replace=True)
        m, _, _ = missions(F, k, sample=smp, seed=SEED + r + 1)
        Hn = ref.components_ / np.linalg.norm(ref.components_, axis=1, keepdims=True)
        Gn = m.components_ / np.linalg.norm(m.components_, axis=1, keepdims=True)
        cos = Hn @ Gn.T; ri, ci = linear_sum_assignment(-cos)
        stab.append(dict(k=k, rep=r, matched_cosine=float(cos[ri, ci].mean()), ari=float(adjusted_rand_score(lab0, m.transform(Xn).argmax(1)))))
ST = pd.DataFrame(stab).groupby("k").mean().drop(columns="rep"); ST.to_csv(TAB / "m3_rq4_mission_stability.csv"); print(ST)

# ---- final test on October to December (refit on January to September) ---------------------
F2 = window_features("fit2", L, items); Qt = queries(T2, T3, F2); Mk2 = markov(F2)
m2, Xn2, b2 = missions(F2, kbest)
S_t = taste(F2, Qt); S_m = seq(Mk2, Qt); S_n = mission_scores(F2, Qt, m2)
res = {"Taste only (baseline)": score(Qt, S_t), "Markov, sequence only": score(Qt, S_m),
       f"NMF missions (k={kbest})": score(Qt, S_n), "Taste + Markov": score(Qt, norm(S_t) + lam * norm(S_m))}
rows = []
for name, d in res.items():
    e1, _ = household_bootstrap(d, ["recall"]); dx = d[d.has_explore]; e2, _ = household_bootstrap(dx, ["explore_recall"])
    rows.append(dict(model=name, recall=e1.loc["recall", "mean"], recall_lo=e1.loc["recall", "lo"], recall_hi=e1.loc["recall", "hi"],
                     explore_recall=e2.loc["explore_recall", "mean"], explore_lo=e2.loc["explore_recall", "lo"], explore_hi=e2.loc["explore_recall", "hi"],
                     trips=len(d)))
R = pd.DataFrame(rows); R.to_csv(TAB / "m3_rq4_next_trip.csv", index=False); print(R.round(4))
diff = res["Taste + Markov"].assign(base=res["Taste only (baseline)"].recall, base_e=res["Taste only (baseline)"].explore_recall)
diff["d"] = diff.recall - diff.base; diff["de"] = diff.explore_recall - diff.base_e
gd, _ = household_bootstrap(diff, ["d"]); ge, _ = household_bootstrap(diff[diff.has_explore], ["de"])
gain = dict(recall_gain=float(gd.loc["d", "mean"]), recall_gain_lo=float(gd.loc["d", "lo"]), recall_gain_hi=float(gd.loc["d", "hi"]),
            explore_gain=float(ge.loc["de", "mean"]), explore_gain_lo=float(ge.loc["de", "lo"]), explore_gain_hi=float(ge.loc["de", "hi"]), lambda_=float(lam))
print(gain)
prof = []
share = np.bincount(m2.transform(Xn2).argmax(1), minlength=kbest) / Xn2.shape[0]
for j, h in enumerate(m2.components_):
    top = np.argsort(-h)[:6]
    prof.append(dict(mission=j + 1, basket_share=float(share[j]), top_categories="; ".join(items[top].str.lower())))
PR = pd.DataFrame(prof).sort_values("basket_share", ascending=False); PR.to_csv(TAB / "m3_rq4_mission_profiles.csv", index=False); print(PR.to_string())

fig, axs = plt.subplots(1, 2, figsize=(13.5, 4.6))
cols = [GREY, BLUE, GREEN, ORANGE]; x = np.arange(len(R))
axs[0].bar(x - 0.2, R.recall, 0.4, color=cols, label="All categories in the next trip")
axs[0].bar(x + 0.2, R.explore_recall, 0.4, color=cols, alpha=0.45, label="Categories new to the household")
axs[0].errorbar(x - 0.2, R.recall, yerr=[R.recall - R.recall_lo, R.recall_hi - R.recall], fmt="none", color=MUTED)
axs[0].set_xticks(x); axs[0].set_xticklabels([n.replace(" (", "\n(").replace(", ", "\n") for n in R.model], fontsize=10.5)
axs[0].set(ylabel="Recall@10", ylim=(0, 0.5), title="Next-trip prediction (test: October to December)"); axs[0].legend(fontsize=10.5, loc="upper right")
axs[1].plot(ST.index, ST.matched_cosine, marker="o", color=BLUE, label="Matched profile cosine")
axs[1].plot(ST.index, ST.ari, marker="o", color=ORANGE, label="Adjusted Rand index of basket labels")
axs[1].axvline(kbest, color=MUTED, ls="--", lw=1); axs[1].set(xlabel="Number of missions k", ylim=(0, 1.02), title="Mission stability across bootstrap refits")
axs[1].legend(fontsize=11)
fig.suptitle("Figure 15. RQ4: sequence adds little beyond household taste", x=0.01, ha="left", fontsize=14.5, y=1.02)
fig.tight_layout(); fig.savefig(FIG / "fig15_rq4_sequence.png"); plt.close(fig)
m3_numbers("rq4", dict(k_best=kbest, results=R.set_index("model").to_dict("index"), gain_taste_plus_markov=gain,
                       stability=ST.to_dict("index")))
