"""Step 16. The routed table: the project's final answer to "which association gets which nudge".

Pair level (category A with category B, fitted on January to September):
  bundle       beats chance (FDR 5%) in January to September AND in July to September, partner is
               not a staple, and the RQ5 logistic model gives it under a 50% chance of failing
  leave alone  frequent pairs that do not pass (popularity co-occurrence, staples, fragile rules)
Category level:
  reminder     categories where many households buy on a regular cycle (RQ3); cadence = median gap
Validation: how many of each route still beat chance in October to December (never used to route),
compared with the standard "top pairs by co-occurrence count" list of the same size.
"""
import numpy as np, pandas as pd
from m3lib import OUT, TAB, FIG, T1, T2, T3, m3_numbers, between, pair_stats, pair_frame, plt, BLUE, ORANGE, GREEN, GREY, MUTED
from features import load, window_features

L, items = load(); F = window_features("fit2", L, items)
dept = pd.Series(F["dept"], index=items)
staple = set(items[list(F["staples"][0.05])])
fit = pair_frame(F["S"], items, 10)
q3 = pair_frame(pair_stats(between(L, T1, T2), items, name="q3"), items)[["a", "b", "sig_pos"]].rename(columns={"sig_pos": "sig_q3"})
q4 = pair_frame(pair_stats(between(L, T2, T3), items, name="q4"), items)[["a", "b", "sig_pos", "ratio"]].rename(columns={"sig_pos": "sig_q4", "ratio": "ratio_q4"})
rel = pd.read_csv(TAB / "m3_rq5_test_rules.csv")[["a", "b", "p_Logistic regression"]].rename(columns={"p_Logistic regression": "p_fail"})
R = fit.merge(q3, on=["a", "b"], how="left").merge(q4, on=["a", "b"], how="left").merge(rel, on=["a", "b"], how="left")
R[["sig_q3", "sig_q4"]] = R[["sig_q3", "sig_q4"]].fillna(False).astype(bool)
R["cross_department"] = R.a.map(dept) != R.b.map(dept)
R["has_staple"] = R.a.isin(staple) | R.b.isin(staple)
R["route"] = np.where(R.sig_pos & R.sig_q3 & ~R.has_staple & (R.p_fail < 0.5), "bundle", "leave alone")
R["why"] = np.select([R.route == "bundle", ~R.sig_pos, R.has_staple, ~R.sig_q3, R.p_fail.isna() | (R.p_fail >= 0.5)],
                     ["passes every test", "within chance (popularity co-occurrence)", "involves a staple",
                      "not above chance in the latest quarter", "predicted fragile (RQ5)"], "")
R = R.sort_values(["route", "strength"], ascending=[True, False])
R.to_csv(TAB / "m3_routed_pairs.csv", index=False)

nb = int((R.route == "bundle").sum())
fbt = fit.nlargest(nb, "n").merge(q4, on=["a", "b"], how="left")
val = dict(bundle_pairs=nb, bundle_cross_department=int((R.route == "bundle").sum() and R[R.route == "bundle"].cross_department.sum()),
           bundle_hold_q4=float(R[R.route == "bundle"].sig_q4.mean()),
           frequency_top_same_size_hold_q4=float(fbt.sig_q4.fillna(False).astype(bool).mean()),
           leave_alone_pairs=int((R.route == "leave alone").sum()),
           leave_alone_hold_q4=float(R[R.route == "leave alone"].sig_q4.mean()),
           why_counts=R[R.route == "leave alone"].why.value_counts().to_dict())

# reminders from RQ3 (category regularity in the test quarter's predictions and M2's regularity table)
rq3 = pd.read_csv(OUT / "rq3_test_predictions.csv.gz")
cat = rq3.groupby("label").agg(households=("member", "nunique"), regular_share=("cv", lambda c: (c <= 0.5).mean()),
                               median_gap=("gap_med", "median"))
cat = cat[cat.households >= 100].sort_values("regular_share", ascending=False)
cat["route"] = np.where(cat.regular_share >= 0.15, "reminder", "no timed nudge")
cat.to_csv(TAB / "m3_routed_categories.csv")
val["reminder_categories"] = int((cat.route == "reminder").sum())
val["top_reminders"] = cat[cat.route == "reminder"].head(8).round(2).reset_index().to_dict("records")
top_b = R[(R.route == "bundle") & R.cross_department].head(15)[["a", "b", "n", "ratio", "strength", "p_fail", "sig_q4"]]
top_b.to_csv(TAB / "m3_routed_top_bundles.csv", index=False)
val["top_cross_department_bundles"] = top_b.round(3).to_dict("records")
print(val)

fig, axs = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw=dict(width_ratios=[1.5, 1]))
colmap = {"bundle": ORANGE, "leave alone": GREY}
for r, g in R.groupby("route"):
    axs[0].scatter(g.n, g.strength, s=6 if r != "bundle" else 9, color=colmap[r], alpha=0.5 if r != "bundle" else 0.8, linewidths=0, label=r)
axs[0].set(xscale="log", yscale="log", xlabel="Baskets with both categories (January to September)", ylabel="Strength above chance",
           title="Every frequent pair, routed"); axs[0].axhline(1, color=MUTED, ls="--", lw=1); axs[0].legend(markerscale=3)
v = [val["bundle_hold_q4"], val["frequency_top_same_size_hold_q4"], val["leave_alone_hold_q4"]]
axs[1].bar(range(3), v, color=[ORANGE, BLUE, GREY])
for i, x in enumerate(v): axs[1].text(i, x + 0.01, f"{x:.0%}", ha="center", fontsize=11)
axs[1].set_xticks(range(3)); axs[1].set_xticklabels(["Routed to\nbundle", f"Top {nb} by\nco-occurrence", "Leave\nalone"])
axs[1].set(ylim=(0, 1.08), ylabel="Still above chance in October to December", title="Held-out check of the routing")
fig.suptitle("Figure 17. The routed table: which associations earn a bundle slot", x=0.01, ha="left", fontsize=14.5, y=1.02)
fig.tight_layout(); fig.savefig(FIG / "fig17_routing.png"); plt.close(fig)
m3_numbers("routing", val)
