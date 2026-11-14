"""Step 4. RQ2 baselines B0 (global popularity) and B1 (household frequency) on the tuning
quarter, under both protocols. They have no hyperparameters, so this only records their tuning-quarter scores.
Under the cross-sell protocol B0 is the signal-free reference every candidate has to beat."""
import pickle, pandas as pd
from m3lib import OUT, TAB
from features import load, window_features
from harness import evaluate, summarize
from rq2lib import b0, b1

L, items = load(); F = window_features("fit", L, items)
Q = pickle.load(open(OUT / "queries_tune.pkl", "rb"))
rows = []
for name, fn in [("B0 popularity", b0(F)), ("B1 household frequency", b1(F))]:
    r, t = evaluate(Q, fn, F); rc, _ = evaluate(Q, fn, F, crosssell=True)
    s, _ = summarize(Q, r, t, F, name, ci=False, cs_ranks=rc); rows.append(s); print(s)
pd.DataFrame(rows).to_csv(TAB / "m3_rq2_tune_baselines.csv", index=False)
