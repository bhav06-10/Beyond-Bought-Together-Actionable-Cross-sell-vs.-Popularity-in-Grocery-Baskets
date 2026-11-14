"""Step 1. Add the Complete Journey companion tables (promotions, demographics, campaigns, coupons)
and the extra columns M3 needs (store, week, quantity, package size).

Writes to data/processed/:
  cj_baskets.csv.gz          one row per basket: member, store, week, timestamp
  m3/cj_loads.csv.gz         per basket and category: units and package-size-adjusted load (RQ3)
  m3/cj_promo_cat.csv.gz     per store, week and category: share of products on display / in the mailer
  m3/cj_demographics.csv     801 households
  m3/cj_coupon_baskets.csv   baskets where a coupon on that category was redeemed
and reports/tables/m3_cleaning_log.csv.
"""
import re
import numpy as np, pandas as pd, pyreadr
from m3lib import CFG, P, OUT, TAB, m3_numbers

raw = P["raw"]; log = []
def note(step, before, after, why):
    log.append(dict(step=step, rows_before=before, rows_after=after, removed=before - after, reason=why))
R = lambda f: list(pyreadr.read_r(raw / f).values())[0]
drop = set(CFG["cj"]["drop_departments"])

for f in ["promotions.rds", "campaigns.rda", "campaign_descriptions.rda", "coupons.rda", "coupon_redemptions.rda", "demographics.rda"]:
    if not (raw / f).exists():
        raise SystemExit(f"Missing data/raw/{f}. Run `make data` first.")

# ---- transactions with store, week, quantity, package size ---------------------------------
t = R(CFG["sources"]["cj_transactions"]["file"]); p = R(CFG["sources"]["cj_products"]["file"])
n0 = len(t)
m = t.merge(p[["product_id", "department", "product_category", "package_size"]], on="product_id", how="inner")
m = m[~m.department.isin(drop) & m.product_category.notna() & (m.quantity > 0) & (m.sales_value > 0)]
note("same filters as M2 clean.py (product match, service departments, voids)", n0, len(m), "identical to M2 so baskets match")
m = m.rename(columns={"household_id": "member", "basket_id": "basket", "product_category": "label", "store_id": "store"})
m["basket"] = m.basket.astype(str); m["member"] = m.member.astype(str); m["store"] = m.store.astype(str)

B = m.groupby("basket").agg(member=("member", "first"), store=("store", "first"), week=("week", "first"),
                            transaction_timestamp=("transaction_timestamp", "min")).reset_index()
B.to_csv(P["processed"] / "cj_baskets.csv.gz", index=False)

# package size -> one number per unit family, so "bought the big one" can be measured (RQ3)
UNIT = {"OZ": 1, "FZ": 1, "LB": 16, "GA": 128, "GAL": 128, "QT": 32, "PT": 16, "LTR": 33.8, "L": 33.8, "ML": 0.0338,
        "CT": 1, "PK": 1, "SQ FT": 1, "FT": 1, "DZ": 12}
rx = re.compile(r"^\s*([\d.]+)\s*(OZ|FZ|LB|GAL|GA|QT|PT|LTR|ML|L|CT|PK|SQ FT|FT|DZ)\b")
def size(s):
    if not isinstance(s, str):
        return np.nan
    mm = rx.match(s.upper())
    if not mm:
        return np.nan
    try:
        return float(mm.group(1)) * UNIT[mm.group(2)]
    except ValueError:
        return np.nan
sizes = pd.Series({ps: size(ps) for ps in m.package_size.dropna().unique()})
m["size"] = m.package_size.map(sizes)
m["load"] = m.quantity * m["size"]
loads = m.groupby(["basket", "label"]).agg(units=("quantity", "sum"), load=("load", "sum"),
                                          size_known=("size", lambda s: s.notna().mean())).reset_index()
loads.loc[loads.size_known < 1, "load"] = np.nan          # only trust a load when every line had a size
loads.drop(columns="size_known").to_csv(OUT / "cj_loads.csv.gz", index=False)

# ---- promotions -> category level ------------------------------------------------------------
pr = R("promotions.rds")
n1 = len(pr)
pr = pr.merge(p[["product_id", "department", "product_category"]], on="product_id", how="inner")
pr = pr[~pr.department.isin(drop) & pr.product_category.notna()]
note("promotions: join to products, drop service departments", n1, len(pr), "category is the M3 item level")
pr["display"] = pr.display_location.astype(str) != "0"
pr["mailer"] = pr.mailer_location.astype(str) != "0"
pc = pr.groupby(["store_id", "week", "product_category"]).agg(disp_share=("display", "mean"), mail_share=("mailer", "mean"),
                                                              n_promo_products=("display", "size")).reset_index()
pc = pc.rename(columns={"store_id": "store", "product_category": "label"})
pc.to_csv(OUT / "cj_promo_cat.csv.gz", index=False)

# ---- demographics, campaigns, coupons ------------------------------------------------------
dm = R("demographics.rda"); dm["household_id"] = dm.household_id.astype(str)
dm.rename(columns={"household_id": "member"}).to_csv(OUT / "cj_demographics.csv", index=False)
cp = R("coupons.rda").merge(p[["product_id", "product_category"]], on="product_id", how="inner")
cr = R("coupon_redemptions.rda")
cr["member"] = cr.household_id.astype(str); cr["date"] = pd.to_datetime(cr.redemption_date)
red = cr.merge(cp[["coupon_upc", "campaign_id", "product_category"]].drop_duplicates(), on=["coupon_upc", "campaign_id"])
red = red.rename(columns={"product_category": "label"})[["member", "date", "label"]].drop_duplicates()
L = pd.read_csv(P["processed"] / "cj_lines.csv.gz", dtype={"basket": str, "member": str}, parse_dates=["date"])
cb = L.merge(red, on=["member", "date", "label"])[["basket"]].drop_duplicates()
cb.to_csv(OUT / "cj_coupon_baskets.csv", index=False)
camp = R("campaigns.rda")

pd.DataFrame(log).to_csv(TAB / "m3_cleaning_log.csv", index=False)
m3_numbers("data", dict(
    lines=int(len(m)), baskets=int(B.basket.nunique()), stores=int(B.store.nunique()),
    package_size_parsed_share=float(m["size"].notna().mean()),
    promo_rows=int(n1), promo_rows_kept=int(len(pr)), promo_stores=int(pr.store_id.nunique()),
    promo_cat_store_weeks=int(len(pc)), display_share_rows=float(pr.display.mean()), mailer_share_rows=float(pr.mailer.mean()),
    lines_in_promo_stores=float(m.store.isin(set(pr.store_id.astype(str))).mean()),
    demographic_households=int(len(dm)), campaigns=int(camp.campaign_id.nunique()),
    campaign_households=int(camp.household_id.nunique()), coupon_redemptions=int(len(cr)),
    coupon_households=int(cr.member.nunique()), coupon_baskets=int(len(cb))))
print(pd.DataFrame(log).to_string()); print("baskets", B.basket.nunique(), "coupon baskets", len(cb), "size parsed", m["size"].notna().mean())
