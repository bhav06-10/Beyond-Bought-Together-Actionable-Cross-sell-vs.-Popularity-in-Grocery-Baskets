# Data

Nothing in `data/raw/` is committed. `make data` downloads every source it can and checks each
file's MD5 hash against `config.yaml`, so everyone analyses exactly the same bytes. The hashes and
download dates are written to `data/manifest.yaml`.

| Source | Provider | Get it | Licence / terms | Coverage | Size | One row is |
| --- | --- | --- | --- | --- | --- | --- |
| Groceries dataset (`Groceries_dataset.csv`) | Heeral Dedhia, Kaggle | Manual: https://www.kaggle.com/datasets/heeraldedhia/groceries-dataset (free account), or `kaggle datasets download heeraldedhia/groceries-dataset -p data/raw --unzip` | Not stated on the dataset page; we do not redistribute it | Labelled 2014-01-01 to 2015-12-30; no location | 38,765 rows, 1.1 MB | One item on one member's shopping day |
| arules Groceries (`groceries.csv`) | Hahsler, Hornik & Reutterer (2006), shipped with the R package `arules` | Automatic (CSV export hosted on GitHub) | Distributed with arules (GPL-3) | 30 days of point-of-sale data from one local grocery outlet; year and place not published | 9,835 receipts, 0.5 MB | One till receipt (list of item categories) |
| The Complete Journey (`transactions.rds`, `products.rda`) | 84.51° (dunnhumby/Kroger), packaged by the R package `completejourney` | Automatic (package data on GitHub) | Package licensed CC0 | 2,469 frequent-shopper households, 1 Jan 2017 to 1 Jan 2018, US; store locations not given | 1,469,307 product lines, 92,331 products, 13.6 MB | One product line in one household's shopping trip |

`label_map.csv` maps every arules/Kaggle label to a cleaned label and one of 12 departments.
It is the only hand-made input; edit it in a pull request so changes are reviewed.

## Added in Milestone 3

The Complete Journey companion tables come from the same GitHub mirror and are fetched and hash-checked by `make data`. They share the household, product, store and week keys with the transactions, so no new cleaning logic is needed (`src/m3/step01_data.py`).

| Source | Get it | Size | One row is | Used for |
| --- | --- | --- | --- | --- |
| CJ `promotions.rds` | Automatic | 20,940,529 rows, 25 MB | A product in a store-week, with its display and mailer placement | Promotions natural experiment (step 11), C3 feature |
| CJ `demographics.rda` | Automatic | 801 households | Age, income, home ownership, household size, kids | Subgroup evaluation only (step 14) |
| CJ `campaigns.rda`, `campaign_descriptions.rda`, `coupons.rda`, `coupon_redemptions.rda` | Automatic | 27 campaigns, 116,204 coupon-product links, 2,102 redemptions | A targeted campaign, a coupon, or a redemption | Coupon sensitivity check (step 14) |

Optional outside sources. They need a login, so they are downloaded by hand. Every script that uses them skips cleanly when the files are missing.

| Source | Get it | Put in | Used for |
| --- | --- | --- | --- |
| Instacart Market Basket Analysis | https://www.kaggle.com/c/instacart-market-basket-analysis/data (accept the competition rules) | `data/raw/instacart/`: `orders.csv`, `order_products__train.csv`, `order_products__prior.csv` (optional, about 550 MB), `products.csv`, `aisles.csv`, `departments.csv` | Step 15: RQ2 replication, reorder-flag check of RQ3, add-to-cart direction test |
| dunnhumby Carbo-Loading | https://www.dunnhumby.com/source-files/ (free form) | `data/raw/carbo/`: `dh_transactions.csv`, `dh_product_lookup.csv` | Step 9: do the pair measures recover known complements (pasta with sauce, pancake mix with syrup)? |
