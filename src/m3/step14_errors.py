"""Step 14. Error, subgroup and robustness analysis of the RQ2 test results.

  where models fail  Recall@10 by target type, department, basket size and how much history the
                     household has
  substitutes        recommendations that look like the basket in embedding space but co-occur
                     LESS than chance with it (two kinds of the same thing), and their hit rate
  subgroups          Recall@10 by income, household size and kids for the 801 households with
                     demographics (evaluation only; demographics never train a model)
  coupons            does dropping the 972 coupon-redemption baskets change the pair ranking?
  SHAP               LightGBM's built-in SHAP values for C3, read as what the model relies on
  exemplars          clear wins and clear misses of C3 under the cross-sell protocol
"""
import json, pickle
import numpy as np, pandas as pd, lightgbm as lgb
from scipy import stats
from m3lib import OUT, TAB, FIG, SEED, M3, T0, T2, m3_numbers, between, pair_stats, pair_frame, household_bootstrap, plt, BLUE, ORANGE, GREEN, GREY, MUTED
from features import load, window_features
from harness import query_metrics
from rq2lib import b0, b1, c1, c2, c3_features, Promo, FEATURES

K = M3["k"]; rng = np.random.default_rng(SEED)
L, items = load(); F = window_features("fit2", L, items)
Q = pickle.load(open(OUT / "queries_test.pkl", "rb"))
Z = np.load(OUT / "rq2_test_results.npz"); names = list(Z["names"]); ranks = dict(zip(names, Z["ranks"])); tops = dict(zip(names, Z["tops"]))
SHOW = ["B1 household frequency", "C2 null-calibrated naive Bayes", "C3 learning-to-rank hybrid"]
from harness import crosssell_queries
csq = crosssell_queries(Q); csr = dict(zip(names, Z["cs_ranks"])); cst = dict(zip(names, Z["cs_tops"]))
cs = pd.DataFrame({"member": Q["member"][csq]})
for n in names:
    cs[n] = (csr[n] <= K).astype(float)
d = query_metrics(Q, ranks[SHOW[0]])[["member", "warm", "repeat", "explore", "staple"]]
for n in SHOW:
    d[n] = (ranks[n] <= K).astype(float)
d["dept"] = F["dept"][Q["hidden"]]
d["size"] = pd.cut(np.diff(Q["X"].indptr)[Q["row"]], [1, 2, 4, 9, 1000], labels=["2", "3-4", "5-9", "10+"])
trips = np.where(Q["mrow"] >= 0, F["trips"][np.maximum(Q["mrow"], 0)], 0)
d["history"] = pd.qcut(pd.Series(trips).rank(method="first"), 3, labels=["light", "medium", "heavy"]).values
d["target"] = np.select([d.repeat, d.explore & ~d.staple, d.explore & d.staple], ["repeat", "explore, non-staple", "explore, staple"], "cold")
tabs = {}
cs["size"] = d["size"].values[csq]; cs["history"] = d["history"].values[csq]
for col in ["size", "history"]:
    cs.groupby(col, observed=True)[SHOW].mean().to_csv(TAB / f"m3_err_crosssell_by_{col}.csv")
for col in ["target", "size", "history", "dept"]:
    g = d[d.warm].groupby(col, observed=True)[SHOW].mean().assign(queries=d[d.warm].groupby(col, observed=True).size())
    if col == "dept":
        g = g.sort_values("queries", ascending=False).head(10)
    tabs[col] = g; g.to_csv(TAB / f"m3_err_by_{col}.csv")
    print(g.round(3))

# ---- substitutes ranked as complements ----------------------------------------------------
S = F["S"]; ratio = S["O"] / np.maximum(S["E"], 1e-6); cos = F["cos"]
sub = {}
samp = rng.choice(Q["n"], 20000, replace=False)
for n in SHOW:
    flags, hitflags = [], []
    for q in samp:
        ctx = [i for i in Q["X"].indices[Q["X"].indptr[Q["row"][q]]:Q["X"].indptr[Q["row"][q] + 1]] if i != Q["hidden"][q]]
        for j in tops[n][q]:
            a = ctx[int(np.argmax(cos[ctx, j]))]
            f = (cos[a, j] > 0.6) and (ratio[a, j] < 0.8)
            flags.append(f); hitflags.append(f and j == Q["hidden"][q])
    flags = np.array(flags)
    sub[n] = dict(share_of_list=float(flags.mean()), hit_rate_when_flagged=float(np.sum(hitflags) / max(flags.sum(), 1)))
print("substitute-like", sub)
iu = np.triu_indices(len(items), 1)
cand = pd.DataFrame({"a": items[iu[0]], "b": items[iu[1]], "cos": cos[iu], "ratio": ratio[iu], "n": S["O"][iu]})
subs_ex = cand[(cand.cos > 0.6) & (cand.ratio < 0.8) & (cand.n >= 20)].sort_values("cos", ascending=False).head(15)
subs_ex.to_csv(TAB / "m3_err_substitute_examples.csv", index=False)

# ---- subgroups --------------------------------------------------------------------------------
dm = pd.read_csv(OUT / "cj_demographics.csv", dtype={"member": str})
inc = {"Under 15K": "under 35K", "15-24K": "under 35K", "25-34K": "under 35K", "35-49K": "35-74K", "50-74K": "35-74K"}
dm["income_band"] = dm.income.map(inc).fillna("75K and over")
dm["size_band"] = dm.household_size.astype(str).str.replace("5+", "5").astype(int).clip(upper=3).map({1: "1 person", 2: "2 people", 3: "3 or more"})
dm["kids_band"] = np.where(dm.kids_count.astype(str).isin(["0"]), "no kids", "kids")
dd = d[d.warm].merge(dm[["member", "income_band", "size_band", "kids_band"]], on="member")
sg = []
for col in ["income_band", "size_band", "kids_band"]:
    for val, g in dd.groupby(col):
        for n in SHOW:
            e, _ = household_bootstrap(g.rename(columns={n: "h"}), ["h"], reps=500)
            ex = cs[cs.member.isin(set(g.member))]
            sg.append(dict(attribute=col, group=val, model=n, recall=e.loc["h", "mean"], lo=e.loc["h", "lo"], hi=e.loc["h", "hi"],
                           cross_sell_recall=ex[n].mean(), cross_sell_queries=len(ex), households=g.member.nunique(), queries=len(g)))
SG = pd.DataFrame(sg); SG.to_csv(TAB / "m3_err_subgroups.csv", index=False); print(SG.round(3).to_string())

# ---- coupon sensitivity ---------------------------------------------------------------------
cb = set(pd.read_csv(OUT / "cj_coupon_baskets.csv", dtype={"basket": str}).basket)
W = between(L, T0, T2)
S_nc = pair_stats(W[~W.basket.isin(cb)], items, name="fit2_no_coupons")
a = pair_frame(F["S"], items, 10); b = pair_frame(S_nc, items, 10)
m = a.merge(b, on=["a", "b"], suffixes=("", "_nc"))
top_a = set(zip(*a.nlargest(100, "strength")[["a", "b"]].values.T)); top_b = set(zip(*b.nlargest(100, "strength")[["a", "b"]].values.T))
coupon = dict(coupon_baskets_in_window=int(W.basket.isin(cb).sum() and W[W.basket.isin(cb)].basket.nunique()),
              spearman_strength=float(stats.spearmanr(m.strength, m.strength_nc).statistic), top100_overlap=len(top_a & top_b),
              sig_pairs=int(a.sig_pos.sum()), sig_pairs_no_coupons=int(b.sig_pos.sum()))
print("coupons", coupon)

# ---- SHAP for C3 ----------------------------------------------------------------------------
best = json.load(open(OUT / "best_params.json")); p1, p2 = best["C1"], best["C2"]
base = {"b0": b0(F), "b1": b1(F), "c1": c1(F, "fit2", p1["min_support"], p1["min_confidence"]),
        "c2": c2(F, p2["measure"], p2["prior_weight"], p2["clip"])}
booster = lgb.Booster(model_file=str(OUT / "c3_model.txt"))
idx = np.sort(rng.choice(Q["n"], 3000, replace=False))
_, _, Xs, _, _ = c3_features(Q, idx, F, base, Promo(), M3["c3"]["candidates"])
sh = booster.predict(Xs, pred_contrib=True)[:, :-1]
shap = pd.Series(np.abs(sh).mean(0), index=FEATURES).sort_values(ascending=False); shap.rename("mean_abs_shap").to_csv(TAB / "m3_err_c3_shap.csv")
print(shap.round(4))

# ---- exemplars ----------------------------------------------------------------------------------
c3n = "C3 learning-to-rank hybrid"
r3 = csr[c3n]; pos_w = np.nonzero(r3 == 1)[0]; pos_m = np.nonzero(r3 > 50)[0]
wins = rng.choice(pos_w, min(5, len(pos_w)), replace=False); miss = rng.choice(pos_m, min(5, len(pos_m)), replace=False)
rows = []
for kind, ps in [("win", wins), ("miss", miss)]:
    for p in ps:
        q = csq[p]
        ctx = [items[i] for i in Q["X"].indices[Q["X"].indptr[Q["row"][q]]:Q["X"].indptr[Q["row"][q] + 1]] if i != Q["hidden"][q]]
        rows.append(dict(kind=kind, basket="; ".join(c.lower() for c in ctx[:8]) + (" ..." if len(ctx) > 8 else ""),
                         added_new_category=items[Q["hidden"][q]].lower(), c3_rank=int(r3[p]),
                         c3_top3="; ".join(items[j].lower() for j in cst[c3n][p][:3])))
EX = pd.DataFrame(rows); EX.to_csv(TAB / "m3_err_exemplars.csv", index=False); print(EX.to_string())

# ---- Figure 16 -------------------------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(16.5, 4.8))
t = cs.groupby("size", observed=True)[SHOW].mean(); x = np.arange(len(t))
for k, (n, c) in enumerate(zip(SHOW, [GREY, BLUE, ORANGE])):
    axs[0].plot(x, t[n], marker="o", color=c, lw=2, label=n.split(" ")[0])
axs[0].set_xticks(x); axs[0].set_xticklabels(t.index); axs[0].set(xlabel="Categories in the basket", ylabel="Cross-sell Recall@10", title="Cross-sell gets easier with more basket context")
axs[0].legend()
g = SG[SG.attribute == "income_band"]; groups_ = ["under 35K", "35-74K", "75K and over"]
for k, (n, c) in enumerate(zip(SHOW, [GREY, BLUE, ORANGE])):
    gg = g[g.model == n].set_index("group").reindex(groups_)
    xx = np.arange(3) + (k - 1) * 0.22
    axs[1].errorbar(xx, gg.recall, yerr=[gg.recall - gg.lo, gg.hi - gg.recall], fmt="o", color=c, capsize=4, label=n.split(" ")[0])
axs[1].set_xticks(range(3)); axs[1].set_xticklabels(groups_); axs[1].set(ylabel="Overall Recall@10 (95% interval)", title="By household income (801 households)")
axs[1].legend()
top = shap.head(10)[::-1]
axs[2].barh(top.index, top.values, color=ORANGE); axs[2].set(xlabel="Mean |SHAP| (C3 score units)", title="What C3 relies on")
fig.suptitle("Figure 16. Where the recommenders fail, who they serve, and what C3 uses (test quarter)", x=0.01, ha="left", fontsize=14.5, y=1.03)
fig.tight_layout(); fig.savefig(FIG / "fig16_errors.png"); plt.close(fig)
m3_numbers("errors", dict(by_target=tabs["target"].to_dict("index"), by_size=tabs["size"].to_dict("index"),
                          by_history=tabs["history"].to_dict("index"), substitutes=sub, coupons=coupon,
                          shap_top=shap.head(8).to_dict()))
