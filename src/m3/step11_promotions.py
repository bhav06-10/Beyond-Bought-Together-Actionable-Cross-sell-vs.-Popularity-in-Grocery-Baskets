"""Step 11. Promotions as a natural experiment: when lead category A is promoted, does its predicted
partner B show up in more baskets?  Task type: observational causal check (two-way fixed effects).

Candidate pairs   above chance in January to June (FDR 5%), partner B not a staple, ranked by
                  M2 strength; both directions (A promoted -> B outcome)
Control pairs     same partner B, but a lead A' of similar popularity whose pair with B is within
                  chance (ratio 0.9 to 1.1, not significant) -- same outcome, different lead
Outcome period    July to December baskets in every store with promotion records (promotions never used to pick pairs)
Model per pair    y = 1[B in basket] ~ promo_A + promo_B + store effects + week effects (linear
                  probability, fixed effects removed by alternating demeaning), standard errors
                  clustered by store-week. promo_X = mean of display and mailer share of X's products
Comparison        effect for candidates minus effect for their matched controls, paired bootstrap
This is supporting evidence only: the retailer chooses what to promote, so it is not a randomized test.
"""
import numpy as np, pandas as pd
from m3lib import OUT, TAB, FIG, SEED, M3, T1, T3, m3_numbers, between, bmatrix, plt, BLUE, ORANGE, GREY, MUTED
from features import load, window_features

cfg = M3["promo"]; rng = np.random.default_rng(SEED)
L, items = load(); F = window_features("fit", L, items)
from m3lib import pair_frame
T = pair_frame(F["S"], items)
staple = set(items[list(F["staples"][0.05])])
D = pd.concat([T.rename(columns={"a": "A", "b": "B", "prev_a": "pA", "prev_b": "pB"}),
               T.rename(columns={"b": "A", "a": "B", "prev_b": "pA", "prev_a": "pB"})])[["A", "B", "n", "ratio", "strength", "sig_pos", "q", "pA", "pB"]]
D = D[(D.n >= 20) & ~D.B.isin(staple)]
cand = D[D.sig_pos].sort_values("strength", ascending=False)
nulls = D[(D.ratio.between(0.9, 1.1)) & (D.q > 0.5)]
pairs = []
for r in cand.itertuples():
    pool = nulls[(nulls.B == r.B) & (nulls.A != r.A)]
    if len(pool) == 0:
        continue
    c = pool.iloc[np.argmin(np.abs(np.log(pool.pA.values / r.pA)))]
    pairs.append((r.A, r.B, c.A, r.strength))
    if len(pairs) >= cfg["pairs_per_group"]:
        break
pairs = pd.DataFrame(pairs, columns=["A", "B", "A_control", "strength"])

# ---- outcome data: July to December baskets in the busiest stores --------------------------
W = between(L, T1, T3)
promo_stores = set(pd.read_csv(OUT / "cj_promo_cat.csv.gz", dtype={"store": str}, usecols=["store"]).store)
top_stores = W[W.store.isin(promo_stores)].drop_duplicates("basket").store.value_counts().index[:cfg["stores"]]
W = W[W.store.isin(top_stores)]
X, b = bmatrix(W, items); X = X.toarray()
meta = W.drop_duplicates("basket").set_index("basket").loc[b, ["store", "week"]]
pc = pd.read_csv(OUT / "cj_promo_cat.csv.gz", dtype={"store": str})
pc = pc[pc.store.isin(top_stores)]
pc["promo"] = 0.5 * (pc.disp_share + pc.mail_share)
Pm = pc.pivot_table(index=["store", "week"], columns="label", values="promo").reindex(columns=items).fillna(0)
Pb = Pm.reindex(pd.MultiIndex.from_arrays([meta.store.values, meta.week.values])).fillna(0).values
s_idx = pd.factorize(meta.store)[0]; w_idx = pd.factorize(meta.week)[0]
cl = pd.factorize(meta.store.astype(str) + "_" + meta.week.astype(str))[0]


def demean(v, it=12):
    v = v.astype(float).copy()
    for _ in range(it):
        v -= (np.bincount(s_idx, v) / np.bincount(s_idx))[s_idx]
        v -= (np.bincount(w_idx, v) / np.bincount(w_idx))[w_idx]
    return v


cacheP = {}
def dP(j):
    if j not in cacheP: cacheP[j] = demean(Pb[:, j])
    return cacheP[j]


def effect(A, B):
    ia, ib = items.get_loc(A), items.get_loc(B)
    y = demean(X[:, ib]); x = np.column_stack([dP(ia), dP(ib)])
    XtX = x.T @ x
    if np.linalg.cond(XtX) > 1e10:
        return np.nan, np.nan, np.nan
    beta = np.linalg.solve(XtX, x.T @ y); u = y - x @ beta
    g = np.zeros((cl.max() + 1, 2)); np.add.at(g, cl, x * u[:, None])
    V = np.linalg.inv(XtX) @ (g.T @ g) @ np.linalg.inv(XtX)
    base = X[:, ib].mean()
    return beta[0] / base, np.sqrt(V[0, 0]) / base, beta[1] / base   # cross effect (A on B), its SE, own effect (B on B)


rows = []
for r in pairs.itertuples():
    e1, s1, own = effect(r.A, r.B); e0, s0, _ = effect(r.A_control, r.B)
    rows.append(dict(A=r.A, B=r.B, A_control=r.A_control, strength=r.strength, effect=e1, se=s1, effect_control=e0, se_control=s0, own_effect=own))
R = pd.DataFrame(rows).dropna(); R["diff"] = R.effect - R.effect_control
R.to_csv(TAB / "m3_promo_pairs.csv", index=False)
boots = np.array([R["diff"].values[rng.integers(len(R), size=len(R))].mean() for _ in range(M3["bootstrap"])])
med = lambda v: np.median(v)
summary = dict(pairs=int(len(R)), baskets=int(len(b)), stores=int(len(top_stores)),
               mean_effect_candidates=float(R.effect.mean()), mean_effect_controls=float(R.effect_control.mean()),
               mean_diff=float(R["diff"].mean()), diff_lo=float(np.percentile(boots, 2.5)), diff_hi=float(np.percentile(boots, 97.5)),
               share_candidates_positive=float((R.effect > 0).mean()), share_controls_positive=float((R.effect_control > 0).mean()),
               share_pairs_candidate_beats_control=float((R["diff"] > 0).mean()),
               mean_own_promo_effect=float(R.drop_duplicates("B").own_effect.mean()),
               share_own_promo_positive=float((R.drop_duplicates("B").own_effect > 0).mean()),
               note="effect = relative change in partner's basket rate when the lead's promotion share goes from 0 to 1")
print(summary)
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
lo, hi = np.percentile(np.r_[R.effect, R.effect_control], [2, 98])
bins = np.linspace(lo, hi, 40)
axs[0].hist(R.effect_control, bins=bins, color=GREY, alpha=0.8, label="Control lead (no association)")
axs[0].hist(R.effect, bins=bins, histtype="step", lw=2, color=ORANGE, label="Predicted complement lead")
axs[0].axvline(0, color=MUTED, lw=1, ls="--")
axs[0].set(xlabel="Change in partner's basket rate when the lead is fully promoted (relative)", ylabel="Pairs",
           title="Partner response to promoting the lead item"); axs[0].legend(fontsize=11)
axs[1].errorbar([0], [summary["mean_diff"]], yerr=[[summary["mean_diff"] - summary["diff_lo"]], [summary["diff_hi"] - summary["mean_diff"]]],
                fmt="o", color=ORANGE, ms=9, capsize=8, lw=2)
axs[1].axhline(0, color=MUTED, lw=1, ls="--"); axs[1].set_xticks([0]); axs[1].set_xticklabels(["Complement minus control"])
axs[1].set(xlim=(-1, 1), ylabel="Mean difference, 95% bootstrap interval", title=f"{summary['share_pairs_candidate_beats_control']:.0%} of pairs respond more than their control")
fig.suptitle("Figure 13. Do predicted complements respond when the lead item is promoted? (July to December)", x=0.01, ha="left", fontsize=14.5, y=1.02)
fig.tight_layout(); fig.savefig(FIG / "fig13_promotions.png"); plt.close(fig)
m3_numbers("promotions", summary)
