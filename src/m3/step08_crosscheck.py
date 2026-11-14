"""Step 8. Two checks on the RQ2 harness itself.

arules cross-check   the real receipts have no customers or dates, so only B0, C1 and C2 can run;
                     5-fold cross-validation over receipts, same leave-one-out protocol and metrics.
Kaggle placebo       the Kaggle file's members, dates and baskets behave like random labels (M2
                     audit), so no model should beat popularity there under the cross-sell protocol
                     (non-staple targets, staples hidden from every list). This check is what exposed
                     that comparing on non-staple targets without hiding staples favours any model
                     that merely avoids staples. A model that did would mean
                     the harness leaks. Split by date: 2014 fits, 2015 is scored.
"""
import numpy as np, pandas as pd
from m3lib import P, TAB, FIG, SEED, m3_numbers, bmatrix, pair_stats, plt, BLUE, ORANGE, GREEN, GREY, MUTED
import json
from m3lib import OUT
from harness import evaluate, query_metrics
from rq2lib import b0, c1, c2

rng = np.random.default_rng(SEED)
C2P = {k: v for k, v in json.load(open(OUT / "best_params.json"))["C2"].items() if k != "selection"}   # settings tuned on CJ


def window(Lfit, items, name):
    X, _ = bmatrix(Lfit, items); N = X.shape[0]
    pop = np.asarray(X.sum(0)).ravel() / N
    order = np.argsort(-pop)
    return dict(items=items, N=N, pop=pop, staples={q: set(order[: max(1, int(round(q * len(items))))]) for q in [0.05, 0.10]},
                S=pair_stats(Lfit, items, name=name), dept=np.zeros(len(items)))


def queries(Lq, items, F):
    sizes = Lq.groupby("basket").label.size(); Lq = Lq[Lq.basket.isin(sizes.index[sizes >= 2])]
    X, b = bmatrix(Lq, items)
    rows = np.repeat(np.arange(X.shape[0]), np.diff(X.indptr)); hid = X.indices.copy()
    st = np.isin(hid, list(F["staples"][0.05]))
    one = np.ones(len(rows), bool)
    return dict(X=X, row=rows, hidden=hid, member=rows.astype(str), mrow=np.full(len(rows), -1), warm=one, repeat=~one,
                explore=one, staple=st, n=len(rows))


def c1_local(F, Lfit, items, ms, mc):
    """FP-Growth rules fitted on this source's fit data (same scorer as rq2lib.c1)."""
    from mlxtend.frequent_patterns import fpgrowth, association_rules
    X, _ = bmatrix(Lfit, items)
    df = pd.DataFrame(X.toarray().astype(bool))
    fi = fpgrowth(df, min_support=ms, max_len=2)
    I = len(items); C = np.zeros((I, I))
    if len(fi) and (fi.itemsets.apply(len) == 2).any():
        r = association_rules(fi, metric="lift", min_threshold=1.0, num_itemsets=len(df))
        r = r[(r.confidence >= mc)]
        C[[next(iter(a)) for a in r.antecedents], [next(iter(c)) for c in r.consequents]] = r.confidence.values
    from rq2lib import ctx_max
    return lambda Q, idx: ctx_max(Q, idx, C)


def run(Lfit, Lq, items, name):
    F = window(Lfit, items, name); Q = queries(Lq, items, F)
    out = {}
    for mname, fn in {"B0 popularity": b0(F), "C1 association rules": c1_local(F, Lfit, items, 0.002, 0.10),
                      "C2 null-calibrated naive Bayes": c2(F, **C2P)}.items():
        r, _ = evaluate(Q, fn, F); d = query_metrics(Q, r)
        rc, _ = evaluate(Q, fn, F, crosssell=True)        # non-staple targets, staples hidden from every list
        out[mname] = dict(recall_all=d.hit.mean(), recall_crosssell=(rc <= 10).mean(), queries=len(d), crosssell_queries=len(rc))
    return pd.DataFrame(out).T


# ---- arules: 5-fold CV over receipts -------------------------------------------------------
A = pd.read_csv(P["processed"] / "arules_lines.csv.gz", dtype={"basket": str})
ai = pd.Index(sorted(A.label.unique())); bk = A.basket.unique(); rng.shuffle(bk); folds = np.array_split(bk, 5)
res = []
for f, test_b in enumerate(folds):
    tb = set(test_b)
    res.append(run(A[~A.basket.isin(tb)], A[A.basket.isin(tb)], ai, f"arules_cv{f}").assign(fold=f))
AR = pd.concat(res); ARm = AR.groupby(level=0).agg(["mean", "std"]); ARm.columns = ["_".join(c) for c in ARm.columns]
ARm.to_csv(TAB / "m3_rq2_arules_cv.csv"); print(ARm.round(4))

# ---- Kaggle placebo: 2014 fits, 2015 scored ------------------------------------------------
K = pd.read_csv(P["processed"] / "kaggle_lines.csv.gz", dtype={"basket": str, "member": str}, parse_dates=["date"])
ki = pd.Index(sorted(K.label.unique()))
KG = run(K[K.date.dt.year == 2014], K[K.date.dt.year == 2015], ki, "kaggle_2014")
KG.to_csv(TAB / "m3_rq2_kaggle_placebo.csv"); print(KG.round(4))

fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.2))
for ax, (T, title) in zip(axs, [(ARm.rename(columns={"recall_crosssell_mean": "recall_crosssell"}), "arules receipts (5-fold CV): real baskets"),
                                (KG, "Kaggle file (2014 fits, 2015 scored): placebo")]):
    v = T["recall_crosssell"].values
    ax.bar(range(3), v, color=[GREY, GREEN, BLUE])
    for i, x in enumerate(v): ax.text(i, x, f"{x:.1%}", ha="center", va="bottom", fontsize=11)
    ax.set_xticks(range(3)); ax.set_xticklabels(["B0\npopularity", "C1\nrules", "C2\nnaive Bayes"]); ax.set(title=title, ylabel="Recall@10, cross-sell protocol")
fig.suptitle("Figure 10. The same ranking on a second retailer, and no signal where there should be none", x=0.01, ha="left", fontsize=14.5, y=1.03)
fig.tight_layout(); fig.savefig(FIG / "fig10_crosscheck_placebo.png"); plt.close(fig)
m3_numbers("rq2_crosscheck", dict(arules=ARm.to_dict("index"), kaggle_placebo=KG.to_dict("index")))
