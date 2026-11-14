"""Step 17. Redundancy and obviousness control for the C1 association rules (RQ2).

Task type: association / frequent-pattern mining. The rules are the ones C1 uses in the final
test (FP-Growth on January to September, support and confidence from step 5).

  Redundancy   A two-item rule {a, c} -> b is redundant when its confidence does not beat the
               better of its one-item sub-rules (a -> b, c -> b) by at least the minimum
               improvement (Bayardo, Agrawal and Gunopulos 2000). Mirror pairs (a -> b and b -> a)
               are counted as one pattern.
  Obviousness  A rule is "obvious" when its consequent is a staple (top 5% by popularity): it
               would be recommended to almost anyone. A same-department rule is reported too.
  Effect       C1 is re-scored on the test quarter with the redundant rules removed, under both
               protocols, to show whether pruning changes what it recommends.
"""
import json, pickle
import numpy as np, pandas as pd
from m3lib import OUT, TAB, M3, m3_numbers
from features import load, window_features
from harness import evaluate, query_metrics, crosssell_queries
from rq2lib import _rules, ctx_max

MIN_IMPROVEMENT = 0.01
L, items = load(); F = window_features("fit2", L, items)
best = json.load(open(OUT / "best_params.json"))["C1"]
r = _rules("fit2", best["min_support"])
r = r[r.confidence >= best["min_confidence"]].reset_index(drop=True)
one = r[r.ante.apply(len) == 1]
S = F["S"]                                                  # sub-rule confidence from the same window, whether or not it passed the threshold
CONF = np.where(S["prev"][:, None] > 0, S["O"] / np.maximum(S["prev"][:, None], 1), 0)
r["n_ante"] = r.ante.apply(len)
r["best_subrule"] = [max(CONF[a, b] for a in ante) if len(ante) == 2 else np.nan for ante, b in zip(r.ante, r.b)]
r["redundant"] = (r.n_ante == 2) & (r.confidence < r.best_subrule + MIN_IMPROVEMENT)
staples = set(F["staples"][0.05]); dept = F["dept"]
r["obvious_staple"] = r.b.isin(staples)
r["same_dept"] = [all(dept[a] == dept[b] for a in ante) for ante, b in zip(r.ante, r.b)]
pairs1 = {tuple(sorted((a[0], b))) for a, b in zip(one.ante, one.b)}

summary = dict(
    rules=int(len(r)), one_item_rules=int((r.n_ante == 1).sum()), two_item_rules=int((r.n_ante == 2).sum()),
    redundant_two_item=int(r.redundant.sum()), redundant_share_of_two_item=float(r.redundant.sum() / max((r.n_ante == 2).sum(), 1)),
    one_item_distinct_pairs=int(len(pairs1)), mirror_duplicates=int((r.n_ante == 1).sum() - len(pairs1)),
    obvious_staple_consequent_share=float(r.obvious_staple.mean()), same_department_share=float(r.same_dept.mean()),
    kept_after_pruning=int((~r.redundant).sum()), min_improvement=MIN_IMPROVEMENT)


def scorer(rules):
    I = len(F["items"]); C = np.zeros((I, I))
    o = rules[rules.n_ante == 1]; C[[a[0] for a in o.ante], o.b.values] = o.confidence.values
    two = {}
    for a, b, c in zip(rules.ante, rules.b, rules.confidence):
        if len(a) == 2:
            two.setdefault(a, np.zeros(I))[b] = max(two.get(a, np.zeros(I))[b], c)
    def f(Q, idx):
        S = ctx_max(Q, idx, C); X = Q["X"]; cache = {}
        for k, (rw, h) in enumerate(zip(Q["row"][idx], Q["hidden"][idx])):
            if rw not in cache:
                its = sorted(X.indices[X.indptr[rw]:X.indptr[rw + 1]])
                cache[rw] = [(a, two[a]) for a in ((x, y) for i, x in enumerate(its) for y in its[i + 1:]) if a in two]
            for a, v in cache[rw]:
                if h not in a:
                    S[k] = np.maximum(S[k], v)
        return S
    return f


Q = pickle.load(open(OUT / "queries_test.pkl", "rb"))
res = {}
for name, rules in {"C1 all rules": r, "C1 redundant rules removed": r[~r.redundant]}.items():
    rk, _ = evaluate(Q, scorer(rules), F); d = query_metrics(Q, rk)
    rc, _ = evaluate(Q, scorer(rules), F, crosssell=True)
    res[name] = dict(rules=int(len(rules)), recall_all_warm=float(d[d.warm].hit.mean()), recall_crosssell=float((rc <= M3["k"]).mean()))
E = pd.DataFrame(res).T
summary["effect"] = E.to_dict("index")
pd.DataFrame([{k: v for k, v in summary.items() if k != "effect"}]).T.rename(columns={0: "value"}).to_csv(TAB / "m3_rq2_rule_redundancy.csv")
E.to_csv(TAB / "m3_rq2_rule_pruning_effect.csv")
r.assign(ante=r.ante.apply(lambda a: " + ".join(items[i] for i in a)), b=r.b.map(lambda i: items[i])) \
 .sort_values("lift", ascending=False).head(5000).to_csv(TAB / "m3_rq2_rules_annotated.csv", index=False)   # top 5,000 by lift
print(json.dumps(summary, indent=1))
m3_numbers("rq2_rule_redundancy", summary)
