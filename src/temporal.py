"""Time-based structure in Complete Journey (figures 6-8; RQ3, RQ4, RQ5), with the Kaggle file
as a comparison for stability."""
import numpy as np, pandas as pd
from scipy import sparse, stats
import matplotlib.pyplot as plt
from common import CFG, P, BLUE, ORANGE, GREEN, GREY, MUTED, save_numbers
from lib import load_lines, basket_matrix, pair_table

rng = np.random.default_rng(CFG["seed"]); R = CFG["rules"]
cj = load_lines("cj")
res = {}
END = cj.date.max()

# ============ RQ3: repurchase timing ==========================================================
days = cj.drop_duplicates(["member", "label", "date"]).sort_values(["member", "label", "date"])
trips = cj.drop_duplicates(["member", "date"])
trip_days = trips.groupby("member").date.apply(lambda s: np.sort(s.values.astype("datetime64[D]").astype(int)))
n_trips = trip_days.map(len)
vol = cj.groupby(["member", "label"]).n_lines.sum()

g = days.groupby(["member", "label"]).date
first_gap = g.diff().dt.days
days["gap"] = first_gap
h = days.groupby(["member", "label"]).agg(n=("date", "size"), last=("date", "max"), dept=("department", "first"),
                                          gap_mean=("gap", "mean"), gap_sd=("gap", "std"))
h["cv"] = h.gap_sd / h.gap_mean
h["trip_share"] = h.n / n_trips.reindex(h.index.get_level_values(0)).values
h["volume"] = vol.reindex(h.index).values
minp = R["min_purchases_rq3"]
h["kind"] = np.select(
    [h.n < minp, h.trip_share >= 0.6, h.cv <= 0.5],
    ["too few purchases", "bought on most trips", "regular cycle"], "irregular")

# null: same number of purchase days drawn at random from that household's own trip days
elig = h[(h.n >= minp) & (h.trip_share < 0.6)]
null_cv = []
for (m, _), n in zip(elig.index, elig.n.values):
    d = trip_days[m]
    pick = np.sort(rng.choice(d, int(n), replace=False)); gg = np.diff(pick)
    null_cv.append(gg.std(ddof=1) / gg.mean())
null_cv = np.array(null_cv)
obs_cv = elig.cv.values
kinds = h.groupby("kind").agg(histories=("n", "size"), volume=("volume", "sum"))
kinds["share_histories"] = kinds.histories / kinds.histories.sum()
kinds["share_volume"] = kinds.volume / kinds.volume.sum()
kinds.to_csv(P["tables"] / "rq3_history_types.csv")
res["rq3"] = dict(histories=int(len(h)), eligible=int(len(elig)),
                  regular_share_obs=float((obs_cv <= 0.5).mean()), regular_share_null=float((null_cv <= 0.5).mean()),
                  cv_median_obs=float(np.median(obs_cv)), cv_median_null=float(np.median(null_cv)),
                  ks_p=float(stats.ks_2samp(obs_cv, null_cv).pvalue),
                  kinds={k: dict(histories=int(r.histories), share_histories=float(r.share_histories),
                                 share_volume=float(r.share_volume)) for k, r in kinds.iterrows()})
# category-level regularity: which categories are most often on a regular cycle
cat = elig.assign(reg=elig.cv <= 0.5).groupby(level=1).agg(histories=("reg", "size"), regular=("reg", "mean"),
                                                            median_gap=("gap_mean", "median"))
cat = cat[cat.histories >= 200].sort_values("regular", ascending=False)
cat.to_csv(P["tables"] / "rq3_category_regularity.csv")
res["rq3"]["top_regular_categories"] = cat.head(8).round(3).reset_index().to_dict("records")

# Kaplan-Meier: time until the same household buys the same category again
def km(dur, event):
    order = np.argsort(dur); dur, event = dur[order], event[order]
    t = np.unique(dur[event]); s, out = 1.0, []
    at_risk = len(dur) - np.searchsorted(dur, t, side="left")
    deaths = np.bincount(np.searchsorted(t, dur[event]), minlength=len(t))
    for r, dth in zip(at_risk, deaths):
        s *= 1 - dth / r; out.append(s)
    return t, np.array(out)
ev = days.dropna(subset=["gap"])
cens = h.reset_index().assign(dur=lambda d: (END - d["last"]).dt.days)
show = ["FLUID MILK PRODUCTS", "BAKED BREAD/BUNS/ROLLS", "SOFT DRINKS", "CHEESE", "CAT LITTER", "LAUNDRY DETERGENTS"]
show = [c for c in show if c in set(h.index.get_level_values(1))]
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.4))
palette = [BLUE, ORANGE, GREEN, "#eda100", "#e87ba4", "#6250d6"]
medians = {}
for c, col in zip(show, palette):
    e = ev[ev.label == c].gap.values; cd = cens[cens.label == c].dur.values
    dur = np.r_[e, cd].astype(float); evt = np.r_[np.ones(len(e), bool), np.zeros(len(cd), bool)]
    t, s = km(dur, evt)
    medians[c] = float(t[np.argmax(s <= 0.5)]) if (s <= 0.5).any() else None
    axs[0].step(t, s, where="post", color=col, lw=1.8, label=f"{c.lower()} (median {medians[c]:.0f} d)")
axs[0].set(xlim=(0, 120), xlabel="Days since the household last bought it", ylabel="Share not yet bought again",
           title="How long until the same household buys it again")
axs[0].legend(fontsize=11)
bins = np.linspace(0, 3, 61)
axs[1].hist(null_cv, bins=bins, density=True, color=GREY, label="Same household, purchase days drawn at random")
axs[1].hist(obs_cv, bins=bins, density=True, histtype="step", lw=2, color=ORANGE, label="Observed")
axs[1].axvline(0.5, color=MUTED, ls="--", lw=1)
axs[1].set(xlabel="Variation in gaps between purchases (CV; lower = more regular)", ylabel="Density",
           title=f"Regular cycles: {res['rq3']['regular_share_obs']:.0%} observed vs {res['rq3']['regular_share_null']:.0%} by chance")
axs[1].legend(fontsize=11, loc="upper center", bbox_to_anchor=(0.5, -0.2))
fig.suptitle("Figure 6. Repurchase timing in Complete Journey (RQ3)", x=0.01, ha="left", fontsize=14.5, y=1.02)
fig.tight_layout(); fig.savefig(P["figures"] / "fig06_repurchase.png"); plt.close(fig)
res["rq3"]["km_median_days"] = medians

# ============ RQ4: does a trip predict the next trip? ============================================
X, b, items = basket_matrix(cj)
meta = cj.drop_duplicates("basket").set_index("basket").loc[b, ["member", "transaction_timestamp"]].reset_index()
meta["row"] = np.arange(len(meta))
meta = meta.sort_values(["member", "transaction_timestamp"])
def consecutive(order_rows, members):
    same = members[1:] == members[:-1]
    return order_rows[:-1][same], order_rows[1:][same]
rows, mem = meta.row.values, meta.member.values
i, j = consecutive(rows, mem)
O = (X[i].T @ X[j]).toarray()
S1 = np.zeros_like(O, dtype=float); S2 = np.zeros_like(S1)
NR = 30
for _ in range(NR):   # shuffle the order of each household's own trips
    perm = meta.assign(k=rng.random(len(meta))).sort_values(["member", "k"])
    a_, b_ = consecutive(perm.row.values, perm.member.values)
    C = (X[a_].T @ X[b_]).toarray(); S1 += C; S2 += C * C
E = S1 / NR
seq = pd.DataFrame([(items[x], items[y], O[x, y], E[x, y]) for x in range(len(items)) for y in range(len(items)) if x != y],
                   columns=["first", "then", "n", "expected"])
seq = seq[seq.n >= 30].copy()
seq["ratio"] = seq.n / seq.expected
seq["p"] = stats.poisson.sf(seq.n - 1, seq.expected); seq["q"] = stats.false_discovery_control(seq.p)
same = pd.read_csv(P["processed"] / "cj_pairs.csv.gz")
sb = pd.concat([same[["a", "b", "strength", "q"]].rename(columns={"a": "first", "b": "then"}),
                same[["a", "b", "strength", "q"]].rename(columns={"b": "first", "a": "then"})])
sb = sb.rename(columns={"strength": "same_basket", "q": "q_same"})
seq = seq.merge(sb, on=["first", "then"], how="left")
seq["seq_sig"] = (seq.q < R["fdr"]) & (seq.ratio > 1)
seq["same_sig"] = (seq.q_same < R["fdr"]) & (seq.same_basket > 1)
seq.sort_values("ratio", ascending=False).to_csv(P["tables"] / "rq4_next_visit.csv", index=False)
only_next = seq[seq.seq_sig & ~seq.same_sig]
res["rq4"] = dict(consecutive_trip_pairs=int(len(i)), pairs_tested=int(len(seq)), next_visit_sig=int(seq.seq_sig.sum()),
                  next_visit_sig_share=float(seq.seq_sig.mean()), next_only=int(len(only_next)),
                  corr_log=float(np.corrcoef(np.log(seq.ratio), np.log(seq.same_basket.fillna(1)))[0, 1]),
                  median_ratio=float(seq.ratio.median()),
                  top_next_only=only_next.sort_values("ratio", ascending=False).head(8)[["first", "then", "n", "ratio", "same_basket"]].round(3).to_dict("records"))
fig, ax = plt.subplots(figsize=(6.6, 5))
x = np.log2(seq.same_basket.fillna(1)); y = np.log2(seq.ratio)
ax.scatter(x[~seq.seq_sig], y[~seq.seq_sig], s=5, color=GREY, alpha=0.4, linewidths=0, label="Next-visit link within chance")
ax.scatter(x[seq.seq_sig], y[seq.seq_sig], s=7, color=ORANGE, alpha=0.7, linewidths=0, label="Next-visit link above chance (FDR 5%)")
ax.axhline(0, color=MUTED, lw=1, ls="--"); ax.axvline(0, color=MUTED, lw=1, ls="--")
ax.set(xlabel="Same-basket strength, log2 (0 = chance)", ylabel="Next-visit strength, log2 (0 = chance)",
       title="Figure 7. Same-basket pairs vs 'buy A now, B next trip' (RQ4)")
ax.legend(fontsize=11, loc="upper left", markerscale=2)
fig.tight_layout(); fig.savefig(P["figures"] / "fig07_next_visit.png"); plt.close(fig)

# ============ RQ5: do pairs found early still hold later? =========================================
def halves(src, cut):
    L = load_lines(src); out = []
    for part in [L[L.date < cut], L[L.date >= cut]]:
        Xh, _, it = basket_matrix(part); out.append(pair_table(Xh, it))
    m = out[0].merge(out[1], on=["a", "b"], suffixes=("_1", "_2"))
    m = m[(m.n_1 >= R["min_pair_baskets"]) & (m.n_2 >= R["min_pair_baskets"])]
    m["sig_1"] = (m.q_1 < R["fdr"]) & (m.ratio_1 > 1); m["sig_2"] = (m.q_2 < R["fdr"]) & (m.ratio_2 > 1)
    return m
stab = {"cj": halves("cj", pd.Timestamp(CFG["cj"]["split_date"])), "kaggle": halves("kaggle", pd.Timestamp("2015-01-01"))}
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
for ax, (s, m), col, name in zip(axs, stab.items(), [ORANGE, BLUE], ["Complete Journey: Jan–Jun vs Jul–Dec 2017", "Kaggle file: 2014 vs 2015"]):
    rho = stats.spearmanr(m.ratio_1, m.ratio_2).statistic
    surv = m.sig_2[m.sig_1].mean() if m.sig_1.any() else float("nan")
    res.setdefault("rq5", {})[s] = dict(pairs=int(len(m)), spearman=float(rho), sig_first=int(m.sig_1.sum()),
                                        survive=float(surv), new_in_second=int((~m.sig_1 & m.sig_2).sum()))
    ax.scatter(np.log2(m.ratio_1[~m.sig_1]), np.log2(m.ratio_2[~m.sig_1]), s=5, color=GREY, alpha=0.4, linewidths=0, label="Not above chance in first period")
    ax.scatter(np.log2(m.ratio_1[m.sig_1]), np.log2(m.ratio_2[m.sig_1]), s=7, color=col, alpha=0.7, linewidths=0, label="Above chance in first period")
    lim = [-3, 3]; ax.plot(lim, lim, color=MUTED, ls="--", lw=1); ax.set(xlim=lim, ylim=lim)
    ax.set(xlabel="Observed ÷ chance in first period, log2", ylabel="Observed ÷ chance in second period, log2",
           title=f"{name}\nrank correlation {rho:.2f}; {surv:.0%} of early pairs hold up" if surv == surv else f"{name}\nrank correlation {rho:.2f}")
    ax.legend(fontsize=11, loc="upper left", markerscale=2)
fig.suptitle("Figure 8. Do associations found in one period hold in the next? (RQ5)", x=0.01, ha="left", fontsize=14.5, y=1.02)
fig.tight_layout(); fig.savefig(P["figures"] / "fig08_stability.png"); plt.close(fig)

# Kaggle structural break between years (basket sizes)
k = load_lines("kaggle"); ks = k.groupby("basket").agg(n=("label", "size"), y=("date", lambda d: d.iloc[0].year))
res["kaggle_break"] = {int(y): dict(baskets=int(len(g)), mean_size=float(g.n.mean()), share_size2=float((g.n == 2).mean()),
                                    share_5plus=float((g.n >= 5).mean())) for y, g in ks.groupby("y")}
save_numbers("temporal", res)
import json; print(json.dumps(res, indent=1, default=str)[:6000])
