"""Step 3 (build). Create the tuning (Q3) and test (Q4) query sets once and record their size and
the repeat / explore split, which is the core framing of the M3 evaluation."""
import pickle
from m3lib import OUT, T1, T2, T3, m3_numbers
from features import load, window_features
from harness import build_queries

L, items = load()
info = {}
for name, (fw, s, e) in {"tune": ("fit", T1, T2), "test": ("fit2", T2, T3)}.items():
    F = window_features(fw, L, items)
    Q = build_queries(L, items, s, e, F)
    pickle.dump(Q, open(OUT / f"queries_{name}.pkl", "wb"))
    info[name] = dict(baskets=int(Q["X"].shape[0]), households=int(len(set(Q["member"]))), queries=int(Q["n"]),
                      warm_queries=int(Q["warm"].sum()), repeat=int(Q["repeat"].sum()), explore=int(Q["explore"].sum()),
                      explore_nonstaple=int((Q["explore"] & ~Q["staple"]).sum()), cold=int((~Q["warm"]).sum()),
                      repeat_share_of_warm=float(Q["repeat"].sum() / Q["warm"].sum()))
    print(name, info[name])
m3_numbers("harness", info)
