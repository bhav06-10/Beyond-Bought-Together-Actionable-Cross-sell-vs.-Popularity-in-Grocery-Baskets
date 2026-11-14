"""Same-basket structure for every source (figures 3-5, RQ1 and RQ2 tables)."""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from common import CFG, P, BLUE, ORANGE, GREEN, GREY, MUTED, save_numbers
from lib import load_lines, basket_matrix, pair_table, prevalence

R = CFG["rules"]; NAMES = {"kaggle": "Kaggle file", "arules": "arules receipts", "cj": "Complete Journey"}
COL = {"kaggle": GREY, "arules": BLUE, "cj": ORANGE}
data, res = {}, {}
for s in NAMES:
    L = load_lines(s); X, b, items = basket_matrix(L)
    T = pair_table(X, items)
    T.to_csv(P["processed"] / f"{s}_pairs.csv.gz", index=False)
    sizes = np.asarray(X.sum(1)).ravel()
    prev = prevalence(X, items)
    data[s] = dict(L=L, X=X, items=items, T=T, sizes=sizes, prev=prev)
    big = T[T.n >= R["min_pair_baskets"]]
    sig = big[(big.q < R["fdr"]) & (big.ratio > 1)]
    # RQ1: how much of the frequency top-20 survives once chance is accounted for
    top_n = big.nlargest(20, "n"); top_s = big.nlargest(20, "strength")
    res[s] = dict(baskets=int(X.shape[0]), items=int(len(items)), mean_size=float(sizes.mean()),
                  median_size=float(np.median(sizes)), share_single=float((sizes == 1).mean()),
                  pairs_ranked=int(len(big)), pairs_sig_pos=int(len(sig)), share_sig_pos=float(len(sig) / len(big)),
                  alpha=float(T.attrs["alpha"]),
                  top20_count_sig=int(((top_n.q < R["fdr"]) & (top_n.ratio > 1)).sum()),
                  top20_overlap=int(len(set(zip(top_n.a, top_n.b)) & set(zip(top_s.a, top_s.b)))),
                  lift_gt1_but_below_chance=float(((big.lift > 1) & (big.ratio < 1)).mean()),
                  top_item=prev.index[0], top_item_share=float(prev.iloc[0]),
                  top5pct_items_share_of_lines=float(prev.iloc[:max(1, int(len(prev) * 0.05))].sum() / prev.sum()))

# ---- Figure 3: observed vs chance ---------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, s in zip(axs, NAMES):
    T = data[s]["T"]; big = T[T.n >= R["min_pair_baskets"]]
    sig = (big.q < R["fdr"]) & (big.ratio > 1)
    ax.scatter(big.expected[~sig], big.n[~sig], s=6, color=GREY, alpha=0.5, linewidths=0, label="Within chance")
    ax.scatter(big.expected[sig], big.n[sig], s=6, color=COL[s] if s != "kaggle" else BLUE, alpha=0.7, linewidths=0,
               label="Above chance (FDR 5%)")
    lim = [big.expected.min() * 0.8, big.n.max() * 1.2]
    ax.plot(lim, lim, color=MUTED, lw=1, ls="--")
    ax.set(xscale="log", yscale="log", xlabel="Baskets expected by chance", ylabel="Baskets observed",
           title=f"{NAMES[s]}: {res[s]['share_sig_pos']:.0%} of pairs beat chance")
    ax.legend(loc="upper left", markerscale=2)
fig.suptitle("Figure 3. How often item pairs appear together, against a swap-randomized baseline", x=0.01, ha="left", fontsize=14.5, y=1.02)
fig.tight_layout(); fig.savefig(P["figures"] / "fig03_pairs_vs_chance.png"); plt.close(fig)

# ---- Figure 4: basket sizes and item prevalence -------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
for s in NAMES:
    sz = pd.Series(data[s]["sizes"]).clip(upper=20).value_counts(normalize=True).sort_index()
    axs[0].plot(sz.index, sz.values, marker="o", ms=3, lw=1.8, color=COL[s], label=NAMES[s])
    pv = data[s]["prev"].values
    axs[1].plot(np.arange(1, len(pv) + 1) / len(pv) * 100, pv * 100, lw=1.8, color=COL[s], label=NAMES[s])
axs[0].set_xticks([1, 2, 3, 4, 5, 10, 15, 20]); axs[0].set_xticklabels(["1", "2", "3", "4", "5", "10", "15", "20+"])
axs[0].set(xlabel="Distinct items in a basket", ylabel="Share of baskets",
           title="Basket size: the Kaggle file is unnaturally uniform"); axs[0].legend()
axs[1].set(xscale="log", xlabel="Item rank, as % of all items (log scale)", ylabel="% of baskets containing the item",
           title="A few staples sit in a large share of baskets"); axs[1].legend()
for s in ["arules", "cj"]:
    axs[1].annotate(data[s]["prev"].index[0].lower(), (100 / len(data[s]["prev"]), data[s]["prev"].iloc[0] * 100),
                    xytext=(8, 4), textcoords="offset points", fontsize=11, color=MUTED)
fig.suptitle("Figure 4. Basket size and item prevalence across the three sources", x=0.01, ha="left", fontsize=14.5, y=1.02)
fig.tight_layout(); fig.savefig(P["figures"] / "fig04_sizes_prevalence.png"); plt.close(fig)

# ---- Figure 5: the most frequent pairs, judged by lift vs by strength above chance (RQ1) ----
fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))
for ax, s in zip(axs, ["arules", "cj"]):
    big = data[s]["T"][data[s]["T"].n >= R["min_pair_baskets"]]
    top = big.nlargest(12, "n").iloc[::-1]
    y = np.arange(len(top))
    good = ((top.q < R["fdr"]) & (top.ratio > 1)).values
    ax.scatter(top.lift, y, s=40, facecolors="none", edgecolors=MUTED, lw=1.2, label="Classic lift", zorder=3)
    ax.scatter(top.strength, y, s=46, color=np.where(good, COL[s], GREY), zorder=4, label="Strength above chance")
    for yi, (l, st) in enumerate(zip(top.lift, top.strength)):
        ax.plot([l, st], [yi, yi], color="#d6d5cf", lw=1, zorder=2)
    ax.axvline(1, color=MUTED, ls="--", lw=1)
    ax.set_yticks(y); ax.set_yticklabels([f"{a.lower()} + {b.lower()}  ({int(n):,})" for a, b, n in zip(top.a, top.b, top.n)], fontsize=11)
    ax.set(xlabel="1.0 = no association", title=f"{NAMES[s]}: 12 most frequent pairs (baskets in brackets)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=11)
fig.suptitle("Figure 5. Lift calls almost every frequent pair an association; against chance, many are not", x=0.01, ha="left", fontsize=14.5, y=1.02)
fig.tight_layout(); fig.savefig(P["figures"] / "fig05_popularity_vs_strength.png"); plt.close(fig)

# ---- RQ2 preview: actionable candidates --------------------------------------------------
for s in ["arules", "cj"]:
    d = data[s]; T = d["T"]; L = d["L"]
    n_staple = max(1, int(round(len(d["prev"]) * R["staple_top_share"])))
    staples = set(d["prev"].index[:n_staple])
    dept = L.drop_duplicates("label").set_index("label").department
    c = T[(T.n >= R["min_pair_baskets"]) & (T.q < R["fdr"]) & (T.ratio > 1)].copy()
    c = c[~(c.a.isin(staples) & c.b.isin(staples))]
    c["cross_department"] = c.a.map(dept) != c.b.map(dept)
    c["conf_a_to_b"] = c.n / c.prev_a; c["conf_b_to_a"] = c.n / c.prev_b
    c = c.sort_values("strength", ascending=False)
    c.to_csv(P["tables"] / f"rq2_candidates_{s}.csv", index=False)
    res[s].update(staples=sorted(staples), rq2_candidates=int(len(c)), rq2_cross_dept=int(c.cross_department.sum()))
    print(s, "candidates", len(c), "cross-dept", int(c.cross_department.sum()))
    print(c.head(12)[["a", "b", "n", "expected", "strength", "lift", "kulc", "cross_department"]].round(2).to_string())
save_numbers("pairs", res)
for s in res: print(s, {k: v for k, v in res[s].items() if k != "staples"})
