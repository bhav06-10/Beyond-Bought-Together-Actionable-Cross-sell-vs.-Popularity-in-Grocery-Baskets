"""Step 9. RQ1: which pair measure best predicts how strong a pair will be NEXT period?
Task type: regression / calibration over item pairs.

Unit      every CJ pair seen in at least 10 baskets in January to June (fit window)
Target    second-half (July to December) log ratio log((n2 + 0.5) / (E2 + 0.5)), with E2 from a
          swap-randomized baseline fitted on second-half baskets only (a label, never a feature)
Baseline  first-half lift carried forward (persistence of the standard measure)
Candidates  M2 swap-null shrunk strength; DuMouchel's Gamma-Poisson Shrinker (EBGM)
Metrics   MAE (plain and count-weighted), Spearman, precision of the top 50, and the sign-error
          rate: of pairs a measure calls positive, the share that fall below chance next half.
Cross-checks  arules receipts (random halves, no dates); dunnhumby Carbo-Loading known complements
          when its files are in data/raw/carbo/.
"""
import numpy as np, pandas as pd
from scipy import stats
from m3lib import (CFG, P, TAB, FIG, SEED, T1, T3, m3_numbers, between, pair_stats, pair_frame, plt,
                   BLUE, ORANGE, GREEN, GREY, MUTED)
from features import load, window_features

MEAS = {"lift (baseline)": "lift", "M2 strength": "strength", "EBGM": "ebgm"}
COL = {"lift (baseline)": GREY, "M2 strength": BLUE, "EBGM": ORANGE}
rng = np.random.default_rng(SEED)


def compare(S1, S2, items, min_n=10, label=""):
    a = pair_frame(S1, items); b = pair_frame(S2, items)
    m = a.merge(b[["a", "b", "n", "expected", "ratio", "sig_pos"]], on=["a", "b"], suffixes=("", "_2"), how="left")
    m = m[m.n >= min_n].copy()
    m[["n_2", "expected_2", "ratio_2"]] = m[["n_2", "expected_2", "ratio_2"]].fillna({"n_2": 0, "ratio_2": 0})
    m["expected_2"] = m.expected_2.fillna(m.expected_2.median())
    m["sig_pos_2"] = m.sig_pos_2.fillna(False).astype(bool)
    m["y"] = np.log((m.n_2 + 0.5) / (m.expected_2 + 0.5))
    out = {}
    for name, col in MEAS.items():
        x = np.log(np.maximum(m[col].values, 1e-6)); err = np.abs(x - m.y.values)
        pos = m[col].values > 1
        top = m.nlargest(50, col)
        boots = []
        for _ in range(500):
            i = rng.integers(len(m), size=len(m)); boots.append(err[i].mean())
        out[name] = dict(pairs=int(len(m)), mae=float(err.mean()), mae_lo=float(np.percentile(boots, 2.5)),
                         mae_hi=float(np.percentile(boots, 97.5)), mae_weighted=float(np.average(err, weights=m.n)),
                         spearman=float(stats.spearmanr(x, m.y).statistic), called_positive=int(pos.sum()),
                         sign_error=float((m.ratio_2.values[pos] < 1).mean()) if pos.any() else np.nan,
                         precision_positive=float(m.sig_pos_2.values[pos].mean()) if pos.any() else np.nan,
                         top50_precision=float(top.sig_pos_2.mean()))
    return pd.DataFrame(out).T.assign(source=label), m


# ---- Complete Journey: January to June predicts July to December ---------------------------
L, items = load(); F = window_features("fit", L, items)
S2 = pair_stats(between(L, T1, T3), items, name="h2")
res_cj, M = compare(F["S"], S2, items, label="Complete Journey")

# ---- arules: two random halves of the receipts ------------------------------------------------
A = pd.read_csv(P["processed"] / "arules_lines.csv.gz", dtype={"basket": str})
ai = pd.Index(sorted(A.label.unique())); bk = A.basket.unique(); half = set(rng.choice(bk, len(bk) // 2, replace=False))
Sa = pair_stats(A[A.basket.isin(half)], ai, name="arules_a"); Sb = pair_stats(A[~A.basket.isin(half)], ai, name="arules_b")
res_ar, _ = compare(Sa, Sb, ai, min_n=5, label="arules (random halves)")

# ---- Carbo-Loading: do the measures recover known complements? ------------------------------
carbo = P["raw"] / "carbo"
res_cl = None
if (carbo / "dh_transactions.csv").exists():
    tx = pd.read_csv(carbo / "dh_transactions.csv", usecols=["upc", "household", "basket"])
    pl = pd.read_csv(carbo / "dh_product_lookup.csv", usecols=["upc", "commodity", "brand"])
    d = tx.merge(pl, on="upc")
    d["label"] = d.commodity.str.strip().str.lower() + " | " + d.brand.str.strip().str.lower()
    top = d.drop_duplicates(["basket", "label"]).label.value_counts(); keep = top.index[top >= 200]
    d = d[d.label.isin(keep)].drop_duplicates(["basket", "label"]); d["basket"] = d.basket.astype(str)
    ci = pd.Index(sorted(d.label.unique()))
    Sc = pair_stats(d, ci, name="carbo")
    T = pair_frame(Sc, ci)
    com = lambda s: s.split(" | ")[0]
    meal = {"pasta": "dinner", "pasta sauce": "dinner", "pancake mixes": "breakfast", "syrups": "breakfast"}
    T["ca"], T["cb"] = T.a.map(com), T.b.map(com)
    T = T[T.ca != T.cb].copy()                               # same-commodity pairs are substitutes, set aside
    T["complement"] = T.ca.map(meal) == T.cb.map(meal)
    from sklearn.metrics import roc_auc_score
    res_cl = pd.DataFrame({n: dict(pairs=int(len(T)), complements=int(T.complement.sum()),
                                   auc=float(roc_auc_score(T.complement, T[c])),
                                   top20_complement_share=float(T.nlargest(20, c).complement.mean())) for n, c in MEAS.items()}).T
    res_cl.to_csv(TAB / "m3_rq1_carbo_known_complements.csv")
else:
    print("Carbo-Loading files not found in data/raw/carbo/; skipping the known-complement check.")

R = pd.concat([res_cj, res_ar]); R.index.name = "measure"; R.to_csv(TAB / "m3_rq1_measures.csv")
print(R.round(4).to_string()); print(res_cl)

# ---- Figure 9: calibration of each measure ------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
for name, col in MEAS.items():
    x = np.log2(np.maximum(M[col], 1e-6)); y = M.y / np.log(2)
    bins = pd.qcut(x, 12, duplicates="drop")
    g = pd.DataFrame({"x": x, "y": y, "b": bins}).groupby("b", observed=True).mean()
    axs[0].plot(g.x, g.y, marker="o", ms=4, lw=1.8, color=COL[name], label=name)
lim = [-2, 3]; axs[0].plot(lim, lim, color=MUTED, ls="--", lw=1)
axs[0].set(xlim=lim, ylim=[-2, 1.5], xlabel="First-half score, log2", ylabel="Second-half strength above chance, log2",
           title="Calibration: lift overstates every pair; shrunk measures sit on the diagonal")
axs[0].legend()
r = res_cj
xx = np.arange(len(r))
axs[1].bar(xx - 0.2, r.sign_error, 0.4, color=[COL[n] for n in r.index], label="Called positive, below chance next half")
axs[1].bar(xx + 0.2, r.top50_precision, 0.4, color=[COL[n] for n in r.index], alpha=0.45, label="Top 50 still above chance (FDR 5%)")
for i, (se, tp) in enumerate(zip(r.sign_error, r.top50_precision)):
    axs[1].text(i - 0.2, se + 0.01, f"{se:.0%}", ha="center", fontsize=11); axs[1].text(i + 0.2, tp + 0.01, f"{tp:.0%}", ha="center", fontsize=11)
axs[1].set_xticks(xx); axs[1].set_xticklabels(r.index); axs[1].set(ylim=(0, 1.1), ylabel="Share of pairs",
          title="Solid: sign errors (lower is better). Light: top-50 precision")
fig.suptitle("Figure 11. RQ1: which pair measure predicts next-half strength (Complete Journey)", x=0.01, ha="left", fontsize=14.5, y=1.02)
fig.tight_layout(); fig.savefig(FIG / "fig11_rq1_measures.png"); plt.close(fig)
m3_numbers("rq1", dict(cj=res_cj.drop(columns="source").to_dict("index"), arules=res_ar.drop(columns="source").to_dict("index"),
                       carbo=None if res_cl is None else res_cl.to_dict("index")))
