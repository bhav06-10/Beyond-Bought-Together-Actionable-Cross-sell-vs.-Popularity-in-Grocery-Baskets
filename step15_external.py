"""Step 15. Outside replication with Instacart (aisle level). Runs only when the files are in
data/raw/instacart/ (manual Kaggle download; see data/README.md).

  RQ2 replication  B0, C1, C2 (and B1 when order_products__prior.csv is present) on a seeded sample
                   of users, same leave-one-out protocol: prior orders fit, each user's last
                   ("train") order is scored. Without the prior file, a random 80/20 split of the
                   last orders is used and only the non-personal models run.
                   The cross-sell protocol is applied as on Complete Journey (aisles new to the
                   user, staples hidden), with user-bootstrap intervals.
  RQ3 replication  For user-aisles with 3+ prior gaps: is the aisle in the last order? Compares
                   regular vs irregular cycles, "due" vs not, and a timing logistic model against
                   frequency alone (AUC on held-out users)
  Direction test   add_to_cart_order: for pairs above chance, is A put in the cart before B more
                   often than after? Binomial test per pair, FDR 5%
"""
import numpy as np, pandas as pd
from scipy import stats
from m3lib import P, TAB, FIG, SEED, m3_numbers, bmatrix, pair_stats, pair_frame, household_bootstrap, plt, BLUE, ORANGE, GREEN, GREY
from harness import evaluate, query_metrics, crosssell_queries, K
from rq2lib import b0, c2, ctx_max

D = P["raw"] / "instacart"
if not (D / "orders.csv").exists():
    print("Instacart files not found in data/raw/instacart/; skipping step 15 (optional replication)."); raise SystemExit(0)
rng = np.random.default_rng(SEED)
import json; from m3lib import OUT
P2 = json.load(open(OUT / "best_params.json"))["C2"]          # the settings tuned on Complete Journey, unchanged
orders = pd.read_csv(D / "orders.csv")
prod = pd.read_csv(D / "products.csv").merge(pd.read_csv(D / "aisles.csv"), on="aisle_id").merge(pd.read_csv(D / "departments.csv"), on="department_id")
users = rng.choice(orders[orders.eval_set == "train"].user_id.unique(), 20000, replace=False)
O = orders[orders.user_id.isin(users)]
def lines(fname):
    op = pd.read_csv(D / fname)
    op = op[op.order_id.isin(O.order_id)].merge(prod[["product_id", "aisle", "department"]], on="product_id")
    return op.merge(O[["order_id", "user_id", "order_number", "days_since_prior_order"]], on="order_id")
last = lines("order_products__train.csv")
has_prior = (D / "order_products__prior.csv").exists()
prior = lines("order_products__prior.csv") if has_prior else None
agg = lambda x: x.sort_values("add_to_cart_order").groupby(["order_id", "aisle"]).agg(
    user_id=("user_id", "first"), cart_pos=("add_to_cart_order", "min"), reordered=("reordered", "max")).reset_index()
A_last = agg(last).rename(columns={"order_id": "basket", "aisle": "label"}); A_last["basket"] = A_last.basket.astype(str)
if has_prior:
    A_fit = agg(prior).rename(columns={"order_id": "basket", "aisle": "label"}); A_fit["basket"] = A_fit.basket.astype(str); A_q = A_last
else:
    bk = A_last.basket.unique(); te = set(rng.choice(bk, len(bk) // 5, replace=False))
    A_fit, A_q = A_last[~A_last.basket.isin(te)], A_last[A_last.basket.isin(te)]
items = pd.Index(sorted(set(A_fit.label) | set(A_q.label)))
S = pair_stats(A_fit, items, name="instacart_fit")
X, _ = bmatrix(A_fit, items); pop = np.asarray(X.sum(0)).ravel() / X.shape[0]; order = np.argsort(-pop)
F = dict(items=items, pop=pop, S=S, staples={0.05: set(order[: int(round(0.05 * len(items)))])}, dept=np.zeros(len(items)))
sizes = A_q.groupby("basket").label.size(); A_q = A_q[A_q.basket.isin(sizes.index[sizes >= 2])]
Xq, bq = bmatrix(A_q, items); rows = np.repeat(np.arange(Xq.shape[0]), np.diff(Xq.indptr)); hid = Xq.indices.copy()
user_of = A_q.drop_duplicates("basket").set_index("basket").user_id.loc[bq].values[rows]
seen = np.zeros(len(rows), bool)
if has_prior:
    hist = A_fit.groupby("user_id").label.apply(set)
    seen = np.array([items[h] in hist.get(u, set()) for u, h in zip(user_of, hid)])
Q = dict(X=Xq, row=rows, hidden=hid, member=user_of.astype(str), mrow=np.full(len(rows), -1), warm=np.ones(len(rows), bool),
         repeat=seen, explore=~seen, staple=np.isin(hid, list(F["staples"][0.05])), n=len(rows))
conf = np.where(S["prev"][:, None] > 0, S["O"] / np.maximum(S["prev"][:, None], 1), 0); np.fill_diagonal(conf, 0)
models = {"B0 popularity": b0(F), "C1 association rules (pair confidence)": lambda Q, idx: ctx_max(Q, idx, conf),
          "C2 null-calibrated naive Bayes": c2(F, P2["measure"], P2["prior_weight"], P2["clip"])}
if has_prior:
    U = pd.Index(sorted(A_fit.user_id.unique())); fq = np.zeros((len(U), len(items)))
    tr = A_fit.groupby("user_id").basket.nunique()
    np.add.at(fq, (U.get_indexer(A_fit.user_id), items.get_indexer(A_fit.label)), 1); fq /= tr.reindex(U).values[:, None]
    ur = U.get_indexer(user_of)
    models["B1 household frequency"] = lambda Q, idx: fq[ur[idx]] + 1e-3 * pop
if has_prior:
    Q["mrow"] = ur; F["hh_freq"] = fq                      # cross-sell protocol hides every aisle the user already bought
res, cs_hits = {}, {}
for n, fn in models.items():
    r, _ = evaluate(Q, fn, F); d = query_metrics(Q, r)
    row = dict(recall_all=d.hit.mean(), recall_repeat=d[d.repeat].hit.mean() if has_prior else np.nan, queries=len(d))
    if has_prior:
        rc, _ = evaluate(Q, fn, F, crosssell=True); csq = crosssell_queries(Q)
        cs_hits[n] = (rc <= K).astype(float); row.update(crosssell_recall=cs_hits[n].mean(), crosssell_queries=len(csq))
    res[n] = row
R = pd.DataFrame(res).T
if has_prior:                                               # household (user) bootstrap intervals, paired across models
    hb = pd.DataFrame(cs_hits); hb["member"] = Q["member"][crosssell_queries(Q)]
    ci, boots = household_bootstrap(hb, list(cs_hits)); R["crosssell_lo"] = ci.lo; R["crosssell_hi"] = ci.hi
    ref = list(cs_hits).index("B0 popularity")
    for j, n in enumerate(cs_hits):
        dlt = boots[:, j] - boots[:, ref]; R.loc[n, "crosssell_vs_B0_lo"] = np.percentile(dlt, 2.5); R.loc[n, "crosssell_vs_B0_hi"] = np.percentile(dlt, 97.5)
R.to_csv(TAB / "m3_ext_instacart_rq2.csv"); print(R.round(4).to_string())

# ---- RQ3 replication: does purchase timing predict the next order? --------------------------
# For every user-aisle with at least three prior gaps: was the aisle in the user's last order, and
# does that depend on how regular the cycle is and on whether the aisle is "due" (days since the
# last purchase within 25% of the user's usual gap)? Instacart caps days_since_prior_order at 30.
rq3 = None
if has_prior:
    gaps = O.sort_values(["user_id", "order_number"]).copy()
    gaps["day"] = gaps.groupby("user_id").days_since_prior_order.cumsum().fillna(0)
    po = prior.drop_duplicates(["order_id", "aisle"])[["user_id", "aisle", "order_number"]]
    po = po.merge(gaps[["user_id", "order_number", "day"]], on=["user_id", "order_number"]).sort_values(["user_id", "aisle", "day"])
    po["gap"] = po.groupby(["user_id", "aisle"]).day.diff()
    h = po.groupby(["user_id", "aisle"]).agg(n=("gap", "count"), mean=("gap", "mean"), std=("gap", "std"), last=("day", "max"))
    h = h[(h.n >= 3) & (h["mean"] > 0)].copy()
    tday = gaps[gaps.eval_set == "train"].set_index("user_id").day
    h["since"] = tday.reindex(h.index.get_level_values(0)).values - h["last"]
    h["regular"] = (h["std"] / h["mean"]) <= 0.5
    h["due"] = (h.since - h["mean"]).abs() <= 0.25 * h["mean"]
    bought = set(zip(A_last.user_id, A_last.label))
    h["bought"] = [k in bought for k in h.index]
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    X3 = np.column_stack([np.log1p(h.since), np.log1p(h["mean"]), np.log1p(h.since) - np.log1p(h["mean"]), h["std"] / h["mean"], np.log(h.n)])
    us = h.index.get_level_values(0).values; tr_u = rng.random(us.max() + 1) < 0.7; tr = tr_u[us]
    lr3 = LogisticRegression(max_iter=1000).fit(X3[tr], h.bought[tr])
    rate = h.groupby(["regular", "due"]).bought.agg(["mean", "size"])
    rq3 = dict(histories=int(len(h)), base_rate=float(h.bought.mean()),
               rate_by_regular_due={f"regular={a}, due={b}": dict(rate=float(r["mean"]), n=int(r["size"])) for (a, b), r in rate.iterrows()},
               auc_frequency_only=float(roc_auc_score(h.bought[~tr], -h["mean"][~tr])),
               auc_timing_hazard=float(roc_auc_score(h.bought[~tr], lr3.predict_proba(X3[~tr])[:, 1])),
               reordered_flag_share=float(A_last.reordered.mean()))
    print(rq3)

# ---- direction test with add_to_cart_order -----------------------------------------------
T = pair_frame(S, items, 30); sig = T[T.sig_pos]
pos = A_last.pivot_table(index="basket", columns="label", values="cart_pos")
dirs = []
for a, b in zip(sig.a, sig.b):
    if a in pos and b in pos:
        both = pos[[a, b]].dropna()
        if len(both) >= 20:
            k = int((both[a] < both[b]).sum()); n = len(both)
            dirs.append(dict(a=a, b=b, baskets=n, a_first=k / n, p=stats.binomtest(k, n).pvalue))
DR = pd.DataFrame(dirs)
if len(DR):
    DR["q"] = stats.false_discovery_control(DR.p); DR = DR.sort_values("q"); DR.to_csv(TAB / "m3_ext_instacart_direction.csv", index=False)
direction = dict(pairs_tested=int(len(DR)), directional=int((DR.q < 0.05).sum()) if len(DR) else 0,
                 examples=DR.head(8).round(3).to_dict("records") if len(DR) else [])
print(direction)
m3_numbers("instacart", dict(users=int(len(users)), has_prior=has_prior, rq2=R.to_dict("index"), rq3_reorder=rq3, direction=direction))

# ---- Figure 18: outside validation (Instacart, and Carbo-Loading from step 9 when present) ----
cl_path = TAB / "m3_rq1_carbo_known_complements.csv"
panels = (1 if has_prior else 0) + (1 if rq3 else 0) + (1 if cl_path.exists() else 0)
if panels:
    fig, axs = plt.subplots(1, panels, figsize=(5.6 * panels, 4.6)); axs = np.atleast_1d(axs); i = 0
    if has_prior:
        names = ["B0 popularity", "B1 household frequency", "C1 association rules (pair confidence)", "C2 null-calibrated naive Bayes"]
        v = R.loc[names, "crosssell_recall"].values; lo = R.loc[names, "crosssell_lo"].values; hi = R.loc[names, "crosssell_hi"].values
        y = np.arange(len(names))[::-1]; cols = [GREY, "#8c8b85", GREEN, BLUE]
        axs[i].barh(y, v, color=cols, height=0.6); axs[i].errorbar(v, y, xerr=[v - lo, hi - v], fmt="none", color="#222", lw=1)
        for yy, vv in zip(y, v): axs[i].text(vv, yy, f"  {vv:.1%}", va="center", fontsize=11)
        axs[i].set_yticks(y); axs[i].set_yticklabels(["B0", "B1", "C1", "C2"])
        axs[i].set(xlim=(0, v.max() * 1.35), xlabel="Cross-sell Recall@10 (95% user bootstrap)", title="Instacart: RQ2 ranking replicates"); i += 1
    if rq3:
        keys = ["regular=False, due=False", "regular=False, due=True", "regular=True, due=False", "regular=True, due=True"]
        lab = ["irregular,\nnot due", "irregular,\ndue", "regular,\nnot due", "regular,\ndue"]
        vals = [rq3["rate_by_regular_due"][k]["rate"] for k in keys]
        axs[i].bar(range(4), vals, color=[GREY, "#8c8b85", ORANGE, ORANGE], alpha=1)
        for x, vv in enumerate(vals): axs[i].text(x, vv + 0.01, f"{vv:.0%}", ha="center", fontsize=11)
        axs[i].set_xticks(range(4)); axs[i].set_xticklabels(lab)
        axs[i].set(ylim=(0, 0.75), ylabel="Aisle in the next order", title="Instacart: RQ3 timing replicates"); i += 1
    if cl_path.exists():
        cl = pd.read_csv(cl_path, index_col=0)
        axs[i].bar(range(len(cl)), cl.auc, color=[GREY, GREEN, BLUE])
        for x, vv in enumerate(cl.auc): axs[i].text(x, vv + 0.01, f"{vv:.2f}", ha="center", fontsize=11)
        axs[i].set_xticks(range(len(cl))); axs[i].set_xticklabels(["lift", "M2 strength", "EBGM"])
        axs[i].set(ylim=(0.5, 1.0), ylabel="AUC: known complements vs other pairs", title="Carbo-Loading: RQ1 known complements")
    fig.suptitle("Figure 18. Outside validation on data the models were never tuned on", x=0.01, ha="left", fontsize=14.5, y=1.03)
    fig.tight_layout(); fig.savefig(FIG / "fig18_external_validation.png"); plt.close(fig)
