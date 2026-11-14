"""Step 2. Build every model feature from the past only, for both fit windows, and document
which features are used and which are deliberately excluded (reports/tables/m3_features.csv)."""
import pandas as pd
from m3lib import TAB, m3_numbers
from features import load, window_features

L, items = load()
info = {}
for w in ["fit", "fit2"]:
    F = window_features(w, L, items)
    info[w] = dict(baskets=int(F["N"]), households=int(len(F["members"])), start=str(F["start"].date()),
                   end_exclusive=str(F["end"].date()), alpha=float(F["S"]["alpha"]),
                   ebgm_prior=[float(x) for x in F["S"]["ebgm_prior"]])
    print(w, info[w])

rows = [
    ("Popularity", "Share of fit-window baskets containing the category", "B0, C1-C3 backfill, C3", "used"),
    ("Household frequency", "Share of the household's fit-window trips containing the category", "B1, C3, RQ4", "used"),
    ("Swap-null strength", "M2 shrunk observed/chance ratio (gamma prior, curveball swap baseline)", "C2, C3, RQ1, RQ5", "used"),
    ("EBGM", "DuMouchel two-gamma empirical-Bayes geometric mean of observed/chance", "C2, C3, RQ1", "used"),
    ("Rule confidence", "FP-Growth rule confidence, antecedent in the basket", "C1, C3", "used"),
    ("Embedding similarity", "Cosine of 32-d SVD of positive PMI (item2vec equivalent)", "C3, error analysis", "used"),
    ("Days since last bought / usual gap", "Household's own repurchase clock from the fit window", "C3, RQ3", "used"),
    ("Same/cross department", "Whether the candidate shares a department with the basket", "C3, routing", "used"),
    ("Category promotion share", "Share of the category's products on display or in the mailer that store-week", "C3, promotions test", "used (known before the trip)"),
    ("Package-size load", "Units times parsed package size, relative to the household's usual load", "RQ3", "used"),
    ("Demographics", "Age, income, household size, kids (801 households)", "Subgroup evaluation only", "excluded from training: 68% missing, and fairness"),
    ("Coupon discounts", "retail/coupon discount amounts on the line", "none", "excluded: recorded with the purchase, so it leaks the outcome"),
    ("Sales value, price", "Line revenue", "none", "excluded: only known after the purchase"),
    ("Product-level IDs", "92k products", "none", "excluded: M3 works at category level, as M2"),
]
pd.DataFrame(rows, columns=["feature", "definition", "used_by", "status"]).to_csv(TAB / "m3_features.csv", index=False)
m3_numbers("features", info)
