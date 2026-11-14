"""Steps 2-6: parse, standardise labels, build tidy line tables for every source.

Writes data/processed/{kaggle,arules,cj}_lines.csv.gz and reports/tables/cleaning_log.csv,
which records every row count change and why.
"""
import pandas as pd, pyreadr
from common import CFG, P, ROOT, save_numbers

log = []
def note(source, step, rows_before, rows_after, why):
    log.append(dict(source=source, step=step, rows_before=rows_before, rows_after=rows_after,
                    removed=rows_before - rows_after, reason=why))

label_map = pd.read_csv(ROOT / "data" / "label_map.csv")
dept = dict(zip(label_map.raw_label, label_map.department))

# ---- Kaggle -----------------------------------------------------------------------------
k = pd.read_csv(P["raw"] / CFG["sources"]["kaggle"]["file"], dtype=str)
n0 = len(k)
k["member"] = k.Member_number                                   # identifier, never a number
k["date"] = pd.to_datetime(k.Date, format="%d-%m-%Y")           # explicit: day comes first
k["raw_label"] = k.itemDescription
k["label"] = k.itemDescription.str.strip()
ambiguous = int((k.Date.str[:2].astype(int) <= 12).sum())
whitespace = int((k.raw_label != k.label).sum())
k["department"] = k.label.map(dept)
assert k.department.notna().all(), "unmapped Kaggle label"
note("kaggle", "parse dates (DD-MM-YYYY), strip labels", n0, len(k), "nothing removed")
dup = k.duplicated(["member", "date", "label"])
k["n_lines"] = k.groupby(["member", "date", "label"]).label.transform("size")
k = k[~dup]
note("kaggle", "collapse repeated item in same member-date", n0, len(k),
     "same item twice in one basket; kept as n_lines, presence is what mining uses")
k["basket"] = k.member + "|" + k.date.dt.strftime("%Y-%m-%d")
k[["basket", "member", "date", "label", "department", "n_lines"]].to_csv(P["processed"] / "kaggle_lines.csv.gz", index=False)
save_numbers("clean_kaggle", dict(rows_raw=n0, ambiguous_day_le12=ambiguous, ambiguous_share=ambiguous / n0,
                                  whitespace_rows=whitespace, duplicate_rows=int(dup.sum()), rows_clean=len(k),
                                  members=k.member.nunique(), baskets=k.basket.nunique(), labels=k.label.nunique(),
                                  date_min=str(k.date.min().date()), date_max=str(k.date.max().date())))

# ---- arules Groceries (one receipt per line) -----------------------------------------
recs = [l.rstrip("\n").split(",") for l in open(P["raw"] / CFG["sources"]["arules"]["file"]) if l.strip()]
a = pd.DataFrame([(i, lab) for i, r in enumerate(recs) for lab in r], columns=["basket", "raw_label"])
n0 = len(a)
a["label"] = a.raw_label.str.strip()
a = a.drop_duplicates(["basket", "label"])
note("arules", "strip labels, drop repeats within a receipt", n0, len(a), "trailing-space labels merged into their clean form")
a["department"] = a.label.map(dept)
a["basket"] = a.basket.astype(str)
a[["basket", "label", "department"]].to_csv(P["processed"] / "arules_lines.csv.gz", index=False)
save_numbers("clean_arules", dict(rows_raw=n0, rows_clean=len(a), baskets=len(recs), labels=a.label.nunique()))

# ---- Complete Journey ----------------------------------------------------------------
t = pyreadr.read_r(P["raw"] / CFG["sources"]["cj_transactions"]["file"])[None]
p = pyreadr.read_r(P["raw"] / CFG["sources"]["cj_products"]["file"])["products"]
n0 = len(t)
nulls_tx = int(t.isna().sum().sum())
m = t.merge(p[["product_id", "department", "product_category", "brand"]], on="product_id", how="left")
unmatched = int(m.department.isna().sum())
m = m[m.department.notna()]
note("cj", "join transactions to products on product_id", n0, len(m), "product_id not in products table")
n1 = len(m); m = m[~m.department.isin(CFG["cj"]["drop_departments"])]
note("cj", "drop service departments", n1, len(m), "fuel, coupons, postal, etc. are not basket items")
n1 = len(m); m = m[m.product_category.notna()]
note("cj", "drop lines with no product category", n1, len(m), "category is the item level used for mining")
n1 = len(m); m = m[(m.quantity > 0) & (m.sales_value > 0)]
note("cj", "drop zero-quantity or zero-value lines", n1, len(m), "voids and adjustments, not purchases")
m["date"] = m.transaction_timestamp.dt.normalize()
m = m.rename(columns={"household_id": "member", "basket_id": "basket", "product_category": "label"})
cat_lines = m.groupby(["basket", "label"]).size().rename("n_lines")
first = m.drop_duplicates(["basket", "label"]).set_index(["basket", "label"])
first = first.join(cat_lines).reset_index()
note("cj", "collapse products to category per basket", len(m), len(first), "several products of one category count once")
first[["basket", "member", "date", "transaction_timestamp", "label", "department", "n_lines"]].to_csv(
    P["processed"] / "cj_lines.csv.gz", index=False)
save_numbers("clean_cj", dict(rows_raw=n0, nulls_in_transactions=nulls_tx, unmatched_products=unmatched,
                              rows_clean=len(m), category_rows=len(first), households=first.member.nunique(),
                              baskets=first.basket.nunique(), categories=first.label.nunique(),
                              departments=first.department.nunique(),
                              date_min=str(first.date.min().date()), date_max=str(first.date.max().date()),
                              product_rows=len(p), package_size_missing=int(p.package_size.isna().sum())))

pd.DataFrame(log).to_csv(P["tables"] / "cleaning_log.csv", index=False)
print(pd.DataFrame(log).to_string())
