"""Step 10. RQ5: which above-chance rules will FAIL next period?  Task type: binary classification.

Unit      a category pair that beats chance (FDR 5%) in one quarter
Label     1 = it no longer beats chance in the following quarter (a fragile rule)
Split     train on Q1 -> Q2, test on Q3 -> Q4 (time order respected; no pair-level random split)
Features  from the earlier quarter only: log basket count, log chance expectation, log strength,
          log EBGM, both items' prevalence, cross-department, month-to-month volatility of the
          ratio, and how often both items were promoted in the same store-week
Baseline  "every rule survives" (predicts the training failure rate for every rule)
Candidates  L2 logistic regression; LightGBM classifier (both tuned by household-free grouped CV
          on the training quarter, grouped by the first item so a category never sits on both sides)
Metric    PR-AUC for the failing class (the minority the business cares about), plus ROC-AUC and Brier
Extra     store transfer: rules mined in a random half of the stores, checked in the other half
"""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, precision_recall_curve
from m3lib import (OUT, TAB, FIG, SEED, m3_numbers, between, pair_stats, pair_frame, bmatrix, plt, BLUE, ORANGE, GREY, MUTED)
from features import load

L, items = load()
Qs = {f"q{i+1}": (pd.Timestamp(s), pd.Timestamp(e)) for i, (s, e) in enumerate(
    [("2017-01-01", "2017-04-01"), ("2017-04-01", "2017-07-01"), ("2017-07-01", "2017-10-01"), ("2017-10-01", "2018-01-02")])}
dept = L.drop_duplicates("label").set_index("label").department
promo = pd.read_csv(OUT / "cj_promo_cat.csv.gz", dtype={"store": str})
stats_ = {q: pair_stats(between(L, s, e), items, name=q) for q, (s, e) in Qs.items()}
FEAT = ["log_n", "log_expected", "log_strength", "log_ebgm", "log_prev_a", "log_prev_b", "cross_dept", "volatility", "co_promo"]


def volatility(W, pairs):
    """SD across months of log((n+0.5)/(independence expectation+0.5)) for each pair."""
    ia, ib = items.get_indexer(pairs.a), items.get_indexer(pairs.b); out = []
    for _, Wm in W.groupby(W.date.dt.month):
        X, _ = bmatrix(Wm, items); O = (X.T @ X).toarray(); prev = np.diag(O); Nm = X.shape[0]
        out.append(np.log((O[ia, ib] + 0.5) / (prev[ia] * prev[ib] / Nm + 0.5)))
    return np.std(np.vstack(out), axis=0)


def co_promo(s, e, pairs):
    w0, w1 = L[(L.date >= s)].week.min(), L[(L.date < e)].week.max()
    pc = promo[(promo.week >= w0) & (promo.week <= w1)].pivot_table(index=["store", "week"], columns="label", values="disp_share").fillna(0)
    on = (pc > 0.25)
    res = []
    for a, b in zip(pairs.a, pairs.b):
        res.append(float((on[a] & on[b]).mean()) if a in on and b in on else 0.0)
    return np.array(res)


def frame(q_from, q_to):
    a = pair_frame(stats_[q_from], items); b = pair_frame(stats_[q_to], items)
    a = a[a.sig_pos & (a.n >= 10)].copy()
    a = a.merge(b[["a", "b", "sig_pos"]], on=["a", "b"], how="left", suffixes=("", "_next"))
    a["fail"] = (~a.sig_pos_next.fillna(False).astype(bool)).astype(int)
    N = stats_[q_from]["N"]
    a["log_n"] = np.log(a.n); a["log_expected"] = np.log(a.expected); a["log_strength"] = np.log(a.strength)
    a["log_ebgm"] = np.log(a.ebgm); a["log_prev_a"] = np.log(a.prev_a / N); a["log_prev_b"] = np.log(a.prev_b / N)
    a["cross_dept"] = (a.a.map(dept) != a.b.map(dept)).astype(int)
    s, e = Qs[q_from]
    a["volatility"] = volatility(between(L, s, e), a[["a", "b"]].reset_index(drop=True))
    a["co_promo"] = co_promo(s, e, a)
    return a.reset_index(drop=True)


tr = frame("q1", "q2"); te = frame("q3", "q4")
print("train rules", len(tr), "fail rate", tr.fail.mean(), "| test rules", len(te), "fail rate", te.fail.mean(), flush=True)
groups = pd.factorize(tr.a)[0]
gkf = GroupKFold(n_splits=5)


def cv_score(make):
    ps = np.zeros(len(tr))
    for i, j in gkf.split(tr, tr.fail, groups):
        m = make().fit(tr.loc[i, FEAT], tr.fail[i]); ps[j] = m.predict_proba(tr.loc[j, FEAT])[:, 1]
    return average_precision_score(tr.fail, ps)


grid_lr = {C: cv_score(lambda C=C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))) for C in [0.01, 0.1, 1, 10]}
C = max(grid_lr, key=grid_lr.get)
grid_gb = {}
for nl in [7, 15, 31]:
    for lr in [0.03, 0.1]:
        grid_gb[(nl, lr)] = cv_score(lambda nl=nl, lr=lr: lgb.LGBMClassifier(n_estimators=300, num_leaves=nl, learning_rate=lr,
                                                                              min_child_samples=30, random_state=SEED, verbose=-1, n_jobs=1))
nl, lr = max(grid_gb, key=grid_gb.get)
models = {"Every rule survives (baseline)": None,
          "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)),
          "LightGBM": lgb.LGBMClassifier(n_estimators=300, num_leaves=nl, learning_rate=lr, min_child_samples=30, random_state=SEED, verbose=-1, n_jobs=1)}
rows, preds = [], {}
for name, m in models.items():
    p = np.full(len(te), tr.fail.mean()) if m is None else m.fit(tr[FEAT], tr.fail).predict_proba(te[FEAT])[:, 1]
    preds[name] = p
    rows.append(dict(model=name, pr_auc=average_precision_score(te.fail, p), roc_auc=roc_auc_score(te.fail, p) if m is not None else 0.5,
                     brier=brier_score_loss(te.fail, p), test_rules=len(te), test_fail_rate=te.fail.mean()))
R = pd.DataFrame(rows); R.to_csv(TAB / "m3_rq5_reliability.csv", index=False); print(R.round(4))
coef = pd.Series(models["Logistic regression"][-1].coef_[0], index=FEAT).sort_values()
coef.rename("standardized_coef").to_csv(TAB / "m3_rq5_logistic_coefficients.csv"); print(coef.round(3))
te.assign(**{f"p_{k}": v for k, v in preds.items()}).to_csv(TAB / "m3_rq5_test_rules.csv", index=False)

# ---- store transfer -------------------------------------------------------------------------
stores = np.array(sorted(L.store.dropna().unique())); rng = np.random.default_rng(SEED)
sa = set(rng.choice(stores, len(stores) // 2, replace=False))
SA = pair_stats(L[L.store.isin(sa)], items, name="stores_a"); SB = pair_stats(L[~L.store.isin(sa)], items, name="stores_b")
fa = pair_frame(SA, items); fb = pair_frame(SB, items)
ta = fa[fa.sig_pos & (fa.n >= 10)].merge(fb[["a", "b", "sig_pos", "n"]], on=["a", "b"], how="left", suffixes=("", "_b"))
transfer = dict(rules_store_half_a=int(len(ta)), survive_in_half_b=float(ta.sig_pos_b.fillna(False).astype(bool).mean()),
                baskets_a=int(SA["N"]), baskets_b=int(SB["N"]))
print(transfer)

# ---- Figure 10 ----------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
axs[0].axhline(te.fail.mean(), color=GREY, ls="--", lw=2, label=f"Every rule survives (baseline) (PR-AUC {te.fail.mean():.2f})")
for (name, p), c in zip(list(preds.items())[1:], [BLUE, ORANGE]):
    pr, rc, _ = precision_recall_curve(te.fail, p)
    axs[0].plot(rc, pr, color=c, lw=2, label=f"{name} (PR-AUC {average_precision_score(te.fail, p):.2f})")
axs[0].set(xlabel="Recall: share of fragile rules caught", ylabel="Precision", ylim=(0, 1.02),
           title="Spotting rules that fail next quarter (test: Q3 to Q4)"); axs[0].legend(fontsize=11)
axs[1].barh(coef.index, coef.values, color=[ORANGE if v > 0 else BLUE for v in coef.values])
axs[1].axvline(0, color=MUTED, lw=1)
axs[1].set(xlabel="Standardized logistic coefficient (positive = more likely to fail)", title="What makes a rule fragile (features are correlated)")
fig.suptitle("Figure 12. RQ5: predicting which above-chance rules will not hold", x=0.01, ha="left", fontsize=14.5, y=1.02)
fig.tight_layout(); fig.savefig(FIG / "fig12_rq5_reliability.png"); plt.close(fig)
m3_numbers("rq5", dict(train_rules=int(len(tr)), train_fail_rate=float(tr.fail.mean()), test_rules=int(len(te)),
                       test_fail_rate=float(te.fail.mean()), logistic_C=C, lgb_leaves=nl, lgb_lr=lr,
                       cv_pr_auc_logistic=grid_lr, cv_pr_auc_lgb={str(k): v for k, v in grid_gb.items()},
                       results=R.set_index("model").to_dict("index"), coefficients=coef.to_dict(), store_transfer=transfer))
