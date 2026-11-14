"""Step 5. Tune the two association candidates on the tuning quarter (July to September), with
every statistic fitted on January to June. Nothing here touches October to December.

C1 grid: FP-Growth minimum support x minimum rule confidence (rules need lift > 1).
C2 grid: pair measure (M2 strength or EBGM) x weight on the popularity prior x clipping below-chance evidence.
Selection score, the same for every tuned model: mean of Recall@10 on all warm queries (open
protocol) and on the cross-sell protocol, so a model cannot win by serving only one kind of target.
"""
import itertools, json, pickle, pandas as pd
from m3lib import OUT, TAB, M3, m3_numbers
from features import load, window_features
from harness import evaluate, summarize
from rq2lib import c1, c2

L, items = load(); F = window_features("fit", L, items)
Q = pickle.load(open(OUT / "queries_tune.pkl", "rb"))
sel = lambda s: 0.5 * (s["all warm | Recall@10"] + s["cross-sell | Recall@10"])


def run(fn, name):
    r, t = evaluate(Q, fn, F); rc, _ = evaluate(Q, fn, F, crosssell=True)
    s, _ = summarize(Q, r, t, F, name, ci=False, cs_ranks=rc); return s
rows = []
for ms, mc in itertools.product(M3["c1_grid"]["min_support"], M3["c1_grid"]["min_confidence"]):
    s = run(c1(F, "fit", ms, mc), "C1"); s.update(min_support=ms, min_confidence=mc, selection=sel(s))
    rows.append(s); print("C1", ms, mc, round(s["selection"], 4), flush=True)
g = M3["c2_grid"]
for meas, w, clip in itertools.product(g["measure"], g["prior_weight"], g["clip"]):
    s = run(c2(F, meas, w, clip), "C2"); s.update(measure=meas, prior_weight=w, clip=clip, selection=sel(s))
    rows.append(s); print("C2", meas, w, clip, round(s["selection"], 4), flush=True)
T = pd.DataFrame(rows); T.to_csv(TAB / "m3_rq2_tuning_c1_c2.csv", index=False)
best = {}
for m, keys in [("C1", ["min_support", "min_confidence"]), ("C2", ["measure", "prior_weight", "clip"])]:
    b = T[T.model == m].sort_values("selection", ascending=False).iloc[0]
    best[m] = {k: (b[k].item() if hasattr(b[k], "item") else b[k]) for k in keys}
    best[m]["selection"] = float(b.selection)
json.dump(best, open(OUT / "best_params.json", "w"), indent=2)
m3_numbers("tuning_c1_c2", best); print(best)
