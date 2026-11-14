"""Provenance audit of the Kaggle file (figures 1 and 2).

Fingerprints the Kaggle labels against the arules receipts, then tests whether the member and
date fields carry any signal by comparing them with randomized versions of the same data."""
import numpy as np, pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from common import CFG, P, BLUE, ORANGE, GREY, MUTED, save_numbers
from lib import load_lines

rng = np.random.default_rng(CFG["seed"])
k = load_lines("kaggle"); a = load_lines("arules")
raw = pd.read_csv(P["raw"] / CFG["sources"]["kaggle"]["file"], dtype=str)
res = {}

# --- fingerprint -----------------------------------------------------------------------
kc = raw.itemDescription.str.strip().value_counts()
ac = a.label.value_counts()
common = kc.index.intersection(ac.index)
res["kaggle_labels"] = int(len(kc)); res["shared_labels"] = int(len(common))
res["arules_only_labels"] = sorted(set(ac.index) - set(kc.index))
res["freq_corr"] = float(np.corrcoef(np.log(kc[common]), np.log(ac[common]))[0, 1])
res["instance_ratio"] = float(len(raw) / ac.sum())
arules_raw = [l.rstrip("\n").split(",") for l in open(P["raw"] / CFG["sources"]["arules"]["file"]) if l.strip()]
res["arules_trailing_space_labels"] = sorted({x for r in arules_raw for x in r if x != x.strip()})
res["kaggle_trailing_space_labels"] = sorted(raw.itemDescription[raw.itemDescription != raw.itemDescription.str.strip()].unique())

fig, ax = plt.subplots(figsize=(5.6, 4.6))
ax.scatter(ac[common], kc[common], s=12, color=BLUE, alpha=0.7, linewidths=0)
lo, hi = 1, 4000; ax.plot([lo, hi], [lo * res["instance_ratio"], hi * res["instance_ratio"]], color=MUTED, ls="--", lw=1,
                          label=f"Kaggle = {res['instance_ratio']:.2f} × arules")
ax.set(xscale="log", yscale="log", xlabel="Times the item appears in arules Groceries (30 days of real receipts)",
       ylabel="Times the item appears in the Kaggle file",
       title=f"Figure 1. Same {len(common)} items, same popularity (log r = {res['freq_corr']:.3f})")
ax.legend(loc="upper left"); fig.savefig(P["figures"] / "fig01_provenance_fingerprint.png"); plt.close(fig)

# --- members and dates against chance ---------------------------------------------------
bk = k.drop_duplicates("basket")[["basket", "member", "date"]].sort_values(["member", "date"])
dow = bk.date.dt.dayofweek.value_counts().sort_index()
res["weekday_p"] = float(stats.chisquare(dow).pvalue)
ct = pd.crosstab(k.label, k.date.dt.month); ct = ct[ct.sum(axis=1) >= 100]
res["item_month_p"] = float(stats.chi2_contingency(ct).pvalue)
gaps = bk.groupby("member").date.diff().dt.days.dropna().values
span = (k.date.max() - k.date.min()).days + 1
null_gaps = np.concatenate([np.diff(np.sort(rng.choice(span, n, replace=False)))
                            for n in bk.groupby("member").size().values if n > 1])
res["gap_median"] = float(np.median(gaps)); res["gap_median_null"] = float(np.median(null_gaps))
res["gap_ks_p"] = float(stats.ks_2samp(gaps, null_gaps).pvalue)
res["visits_per_member_median"] = float(bk.groupby("member").size().median())

# shuffle the raw rows (duplicates included), the same way random assignment would have made them
raw_member = raw.Member_number.values; raw_basket = (raw.Member_number + "|" + raw.Date).values
raw_label = raw.itemDescription.str.strip().values

def repeat_pairs(labels):
    t = pd.DataFrame({"m": raw_member, "l": labels, "b": raw_basket})
    return int((t.groupby(["m", "l"]).b.nunique() >= 2).sum())   # bought on 2+ different visits
obs = repeat_pairs(raw_label)
null = np.array([repeat_pairs(rng.permutation(raw_label)) for _ in range(CFG["null_permutations"])])
# duplicates: how many same-item-same-basket rows would random assembly create?
def dup_rows(labels):
    return int(pd.DataFrame({"b": raw_basket, "l": labels}).duplicated().sum())
dn = np.array([dup_rows(rng.permutation(raw_label)) for _ in range(CFG["null_permutations"])])
res.update(dup_obs=dup_rows(raw_label), dup_null_mean=float(dn.mean()), dup_null_sd=float(dn.std()))
res.update(repeat_obs=obs, repeat_null_mean=float(null.mean()), repeat_null_sd=float(null.std()),
           repeat_p=float((np.sum(null >= obs) + 1) / (len(null) + 1)))

fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
axs[0].bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], dow.values, color=BLUE, width=0.7)
axs[0].set(ylabel="Baskets", title=f"No weekly rhythm (χ² p = {res['weekday_p']:.2f})")
bins = np.arange(0, 730, 15)
axs[1].hist(null_gaps, bins=bins, density=True, color=GREY, label="Random visit dates")
axs[1].hist(gaps, bins=bins, density=True, histtype="step", color=BLUE, lw=2, label="Kaggle data")
axs[1].set(xlabel="Days between a member's visits", ylabel="Share of gaps",
           title=f"Visit gaps look random (KS p = {res['gap_ks_p']:.2f})"); axs[1].legend()
axs[2].hist(null, bins=20, color=GREY, label="Items shuffled across members")
axs[2].axvline(obs, color=BLUE, lw=2, label=f"Kaggle data: {obs:,}")
axs[2].set(xlabel="Member–item pairs bought on 2+ visits", ylabel="Shuffled runs",
           title="Members repeat items at chance level")
axs[2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=11)
fig.suptitle("Figure 2. In the Kaggle file, members and dates behave like random labels", x=0.01, ha="left", fontsize=14.5, y=1.02)
fig.tight_layout(); fig.savefig(P["figures"] / "fig02_kaggle_members_dates.png"); plt.close(fig)

save_numbers("audit", res)
print(res)
