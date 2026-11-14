"""Step 7. The one-time RQ2 test on October to December. Every statistic is refitted on January to
September with the settings frozen from steps 5 and 6, and all five models score the SAME queries.

Reports Recall@10 and NDCG@10 per target type with 95% household-bootstrap intervals, the paired
differences between models (Holm-corrected), list popularity, coverage and cross-department share.
Per-query results are saved for the error analysis (step 14) and the routed table (step 16).
"""
import json, pickle
import numpy as np, pandas as pd, lightgbm as lgb
from m3lib import OUT, TAB, FIG, M3, SEED, m3_numbers, holm, plt, BLUE, ORANGE, GREEN, GREY, MUTED, INK
from features import load, window_features
from harness import evaluate, summarize, query_metrics
from rq2lib import b0, b1, c1, c2, c3_features, Promo

L, items = load(); F = window_features("fit2", L, items)
Q = pickle.load(open(OUT / "queries_test.pkl", "rb"))
best = json.load(open(OUT / "best_params.json")); p1, p2 = best["C1"], best["C2"]
base = {"b0": b0(F), "b1": b1(F), "c1": c1(F, "fit2", p1["min_support"], p1["min_confidence"]),
        "c2": c2(F, p2["measure"], p2["prior_weight"], p2["clip"])}
booster = lgb.Booster(model_file=str(OUT / "c3_model.txt")); promo = Promo()


def c3(Q, idx):
    qi, cj, X, _, c2s = c3_features(Q, idx, F, base, promo, M3["c3"]["candidates"])
    S = -1e3 + c2s                                          # outside the candidate set: C2 order, after every candidate
    S[qi, cj] = 1e3 + booster.predict(X)
    return S


models = {"B0 popularity": base["b0"], "B1 household frequency": base["b1"], "C1 association rules": base["c1"],
          "C2 null-calibrated naive Bayes": base["c2"], "C3 learning-to-rank hybrid": c3}
rows, hits, ranks, tops, csr, cst = [], {}, {}, {}, {}, {}
for name, fn in models.items():
    ch = 5000 if name.startswith("C3") else 20000
    r, t = evaluate(Q, fn, F, chunk=ch); rc, tc = evaluate(Q, fn, F, chunk=ch, crosssell=True)
    s, d = summarize(Q, r, t, F, name, cs_ranks=rc); rows.append(s); hits[name] = d.hit.values; ranks[name] = r; tops[name] = t
    csr[name] = rc; cst[name] = tc
    print(name, {k: round(v, 4) for k, v in s.items() if "Recall@10" in k and "lo" not in k and "hi" not in k}, flush=True)
R = pd.DataFrame(rows); R.to_csv(TAB / "m3_rq2_test.csv", index=False)
np.savez_compressed(OUT / "rq2_test_results.npz", names=np.array(list(models)), ranks=np.stack([ranks[k] for k in models]),
                    tops=np.stack([tops[k] for k in models]), cs_ranks=np.stack([csr[k] for k in models]),
                    cs_tops=np.stack([cst[k] for k in models]))

# ---- paired differences (household bootstrap), Holm across all comparisons ------------------
from harness import crosssell_queries
d0 = query_metrics(Q, ranks["B0 popularity"])
csq = crosssell_queries(Q)
segs = {"all warm (open protocol)": (d0.warm.values, hits, np.arange(Q["n"])),
        "cross-sell protocol": (np.ones(len(csq), bool), {k: (v <= M3["k"]).astype(float) for k, v in csr.items()}, csq)}
rng = np.random.default_rng(SEED); comp = []
mem_all = pd.factorize(Q["member"])[0]; H = mem_all.max() + 1
idx = rng.integers(H, size=(M3["bootstrap"], H))
for seg, (mask, hh, qpos) in segs.items():
    mem = mem_all[qpos]
    for ref in ["B0 popularity", "B1 household frequency", "C2 null-calibrated naive Bayes"]:
        for name in models:
            if name == ref:
                continue
            diff = (hh[name] - hh[ref])[mask]; mm = mem[mask]
            sums = np.bincount(mm, diff, H); cnt = np.bincount(mm, minlength=H)
            boot = sums[idx].sum(1) / np.maximum(cnt[idx].sum(1), 1)
            p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
            comp.append(dict(segment=seg, model=name, versus=ref, diff=diff.mean(), lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5), p=max(p, 1 / M3["bootstrap"])))
C = pd.DataFrame(comp); C["p_holm"] = holm(C.p.values); C.to_csv(TAB / "m3_rq2_pairwise.csv", index=False)
print(C.round(4).to_string())

# ---- Figure 9: the main comparison -------------------------------------------------------------
cols = [GREY, "#8c8b85", GREEN, BLUE, ORANGE]
seg_show = ["all warm", "repeat", "explore, non-staple"]
fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw=dict(width_ratios=[1, 1, 1]))
for ax, seg in zip(axs[:2], ["repeat", "cross-sell"]):
    v = R[f"{seg} | Recall@10"].values; lo = R[f"{seg} | Recall@10 lo"].values; hi = R[f"{seg} | Recall@10 hi"].values
    y = np.arange(len(R))[::-1]
    ax.barh(y, v, color=cols, height=0.6)
    ax.errorbar(v, y, xerr=[v - lo, hi - v], fmt="none", color=INK, lw=1)
    for yi, vi in zip(y, v): ax.text(vi, yi, f"  {vi:.1%}", va="center", fontsize=11)
    ax.set_yticks(y); ax.set_yticklabels(R.model if ax is axs[0] else [""] * len(R))
    ax.set(xlim=(0, max(v) * 1.35), xlabel="Recall@10 (95% household bootstrap)",
           title={"repeat": "Repeat targets (open protocol)", "cross-sell": "Cross-sell protocol: new, non-staple only"}[seg])
pop = R["list popularity"].values; ex = R["cross-sell | Recall@10"].values
axs[2].scatter(pop, ex, s=120, color=cols, zorder=3)
for n, xx, yy in zip(R.model, pop, ex): axs[2].annotate(n.split(" ")[0], (xx, yy), xytext=(7, 4), textcoords="offset points", fontsize=11)
axs[2].set(xlabel="Mean popularity of the recommended categories", ylabel="Cross-sell Recall@10", title="Less popularity, more cross-sell")
fig.suptitle("Figure 9. RQ2: which recommender finds real cross-sell? (test: October to December 2017, 263,004 queries)", x=0.01, ha="left", fontsize=14.5, y=1.03)
fig.tight_layout(); fig.savefig(FIG / "fig09_rq2_comparison.png"); plt.close(fig)
m3_numbers("rq2_test", dict(results=R.set_index("model").to_dict("index"), pairwise=C.to_dict("records")))
