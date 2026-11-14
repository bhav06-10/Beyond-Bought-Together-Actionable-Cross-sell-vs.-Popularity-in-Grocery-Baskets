[README.md](https://github.com/user-attachments/files/34835214/README.md)
# Beyond Bought-Together

Actionable cross-sell versus popularity in grocery baskets. A semester project for CSCI 5502 Data Mining at the University of Colorado Boulder.

**Website:** https://bhav06-10.github.io/Beyond-Bought-Together-Actionable-Cross-sell-vs.-Popularity-in-Grocery-Baskets/

## Team

- Bhaveeasheshwar Eswaran (Bhaveeash)
- Raghavendra Kumaran (Raghav)

## Project goal

Grocery recommendations ranked by raw co-occurrence resurface staples and blur the line between scheduled replenishment and genuine cross-sell, which wastes scarce recommendation slots and nudges. This project separates actionable cross-sell from popularity and from replenishment, and routes each association to the intervention it warrants, whether that is a bundle, a timed reminder, a next-trip nudge, or nothing at all.

## Research questions

Revised in Milestone 2 after the data audit. RQ2 is the primary question.

1. How much of a frequency-ranked "frequently bought together" list survives a chance baseline that holds basket size and item popularity fixed, and how badly does lift mislead?
2. Which associations are actionable cross-sell, meaning they beat chance, the paired item is not a near-universal staple, and they ideally cross departments, as opposed to popularity-driven co-occurrence?
3. How many household and category purchase histories follow a regular cycle beyond the household's own shopping rhythm, and what share of purchase volume do they cover?
4. Is cross-visit co-purchase driven by sequence (A now, B on the next trip) or by household taste (the same households buy both at any time)?
5. Which associations flagged as actionable in January to June 2017 still hold in July to December?

## Data

The project uses three public sources. Full provenance, licences and download steps are in [`data/README.md`](data/README.md).

| Source | One row is | Size | Used for |
| --- | --- | --- | --- |
| [Kaggle Groceries](https://www.kaggle.com/datasets/heeraldedhia/groceries-dataset) (Heeral Dedhia) | One item on a member's shopping day | 38,765 rows, 3,898 members, 167 items, labelled 2014 to 2015 | Provenance audit, and a control |
| [arules Groceries](https://search.r-project.org/CRAN/refmans/arules/html/Groceries.html) (Hahsler, Hornik, and Reutterer, 2006) | One till receipt | 9,835 receipts, 169 categories, 30 days | RQ1 and RQ2 |
| [The Complete Journey](https://cran.r-project.org/web/packages/completejourney/vignettes/completejourney.html) (84.51°) | One product line in a household's trip | 1.47 million lines, 2,469 households, all of 2017 | RQ1 to RQ5 |

Our Milestone 2 audit found that the Kaggle file is the arules receipts with member numbers and dates that behave like random labels. It is kept as a control, and the customer and time questions use The Complete Journey.

Raw and processed data are never committed. `make data` downloads the two automatic sources and checks every file's MD5 hash against `config.yaml`. The Kaggle CSV has to be downloaded by hand into `data/raw/`.

## Reproduce Milestone 2

```bash
git clone https://github.com/bhav06-10/Beyond-Bought-Together-Actionable-Cross-sell-vs.-Popularity-in-Grocery-Baskets.git
cd Beyond-Bought-Together-Actionable-Cross-sell-vs.-Popularity-in-Grocery-Baskets
python -m venv .venv && source .venv/bin/activate      # Python 3.13
pip install -r requirements.txt
# put Groceries_dataset.csv in data/raw/ (see data/README.md), then:
make all
```

`make all` runs these steps in order:

| Step | Command | Script | Writes |
| --- | --- | --- | --- |
| Download and hash check | `make data` | `src/acquire.py` | `data/raw/*`, `data/manifest.yaml` |
| Clean and build line tables | `make clean_data` | `src/clean.py` | `data/processed/*_lines.csv.gz`, `reports/tables/cleaning_log.csv` |
| Provenance audit of the Kaggle file | `make audit` | `src/audit.py` | Figures 1 and 2 |
| Same-basket structure, RQ1 and RQ2 | `make pairs` | `src/pairs.py` | Figures 3 to 5, `reports/tables/rq2_candidates_*.csv` |
| Time structure, RQ3 to RQ5 | `make temporal` | `src/temporal.py` | Figures 6 to 8, `reports/tables/rq3_*.csv`, `reports/tables/rq4_next_visit.csv` |

Every headline number in the report and on the website is saved to `reports/numbers.json`. A full run takes about five minutes on a laptop. Random steps use the seed in `config.yaml`, so results repeat exactly. `make tidy` removes all generated outputs.

## Milestone 3: modeling and evaluation

Every model is fitted on the past and scored on the future: statistics come from January to June 2017, settings are tuned on July to September, and the final test on October to December is run once, after refitting on January to September with the settings frozen. Each task is its own script in `src/m3/`, and `make m3` runs them in order (about 40 minutes on a laptop, after `make m2`).

### Which data-mining method answers which question

| RQ | Question in M3 | Task type | Baseline | Candidate methods | Headline metric | Script |
| --- | --- | --- | --- | --- | --- | --- |
| RQ2 (primary) | Which method recommends real cross-sell in future baskets? | Ranking (basket completion) | B0 popularity; B1 household frequency | C1 FP-Growth association rules; C2 null-calibrated strength; C3 LightGBM LambdaMART hybrid | Recall@10 on targets new to the household and not staples | `step04` to `step08` |
| RQ1 | Which pair measure predicts next-half strength? | Regression and calibration over pairs | First-half lift | M2 swap-null shrunk strength; DuMouchel's Gamma-Poisson Shrinker (EBGM) | MAE of the next-half log ratio; sign-error rate | `step09_rq1_measures.py` |
| RQ5 | Which above-chance rules fail next quarter? | Binary classification | Every rule survives | L2 logistic regression; LightGBM classifier | PR-AUC for the failing class | `step10_rq5_reliability.py` |
| Actionability | Do predicted complements respond when the lead item is promoted? | Observational causal check | Matched no-association control pairs | Two-way fixed-effects linear probability model per pair | Complement minus control effect | `step11_promotions.py` |
| RQ3 | When will a household rebuy a category? | Time-to-event forecasting | Household's own median gap | Discrete-time hazard: logistic and LightGBM | Concordance index; integrated Brier score; reminders within 7 days | `step12_rq3_survival.py` |
| RQ4 | Does trip order add anything beyond household taste? | Ranking (next trip) and clustering | Household frequency (taste) | First-order Markov chain; NMF shopping missions; taste + Markov | Recall@10 gain from adding sequence; mission stability | `step13_rq4_sequence.py` |

Shared pieces: `m3lib.py` (periods, swap-randomized pair statistics, EBGM, bootstrap), `features.py` (past-only features), `harness.py` (queries and metrics), `rq2lib.py` (the five RQ2 models). Supporting steps: `step01_data.py` (companion tables), `step02_features.py`, `step03_harness.py`, `step14_errors.py` (errors, subgroups, coupons, SHAP, exemplars), `step15_external.py` (Instacart replication of RQ2 and RQ3, and figure 18 with the Carbo-Loading check from step 9), `step16_routing.py` (the routed table).

### Reproduce Milestone 3

```bash
make m2      # Milestone 2 pipeline, builds data/processed
make m3      # every Milestone 3 step; tables in reports/tables/m3_*.csv, figures 9 to 18
```

Seeds, split dates and every hyperparameter grid are in the `m3:` block of `config.yaml`. Headline numbers are written to `reports/numbers.json` under keys starting with `m3_`. The Instacart and Carbo-Loading checks need a manual download (see `data/README.md`) and skip cleanly without it.

### Outside validation

| Check | Data | Result |
| --- | --- | --- |
| RQ2 replication, cross-sell Recall@10 | Instacart, 20,000 users, 134 aisles | B0 29.0%, B1 29.0%, C1 35.1%, C2 35.3% (C2 vs B0 +5.6 to +7.1 points; C1 and C2 tied; C3 not rerun) |
| RQ3 replication, aisle in next order | Instacart, 169,614 user-aisle histories | Regular and due 61% vs 35% for neither; timing AUC 0.70 vs 0.65 for frequency alone |
| Direction in the cart | Instacart add-to-cart order | 394 of 1,493 above-chance pairs have a clear order (FDR 5%); read as app flow, not cause |
| RQ1 known complements | dunnhumby Carbo-Loading | AUC lift 0.85, M2 strength 0.87, EBGM 0.87; top-20 complement share 90% / 100% / 100% |

## Repository structure

```
README.md
requirements.txt    pinned Python dependencies
config.yaml         paths, seed, and thresholds (all paths relative)
Makefile            one command per pipeline stage
.gitignore          keeps data/raw/ and data/processed/ out of git
data/               README.md (provenance and download steps), label_map.csv, manifest.yaml
src/                pipeline code; common.py and lib.py are shared helpers
src/m3/             Milestone 3 modeling and evaluation, one script per step
figures/            figures generated by make all
reports/            milestone2_report.pdf and .docx, numbers.json, tables/
docs/               the project website (GitHub Pages)
docs/figures/       images used by the website
```

## Milestone 2 report

The full write-up covers data provenance, the data dictionary, reproducible cleaning, quality assessment, and the exploratory analysis: [`reports/milestone2_report.pdf`](reports/milestone2_report.pdf) (also as [Word](reports/milestone2_report.docx)). A shorter version is in the Data and Exploration section of the website.

## Milestone roadmap

- Milestone 0: project idea and framing, approved.
- Milestone 1: framing, research questions, team, website, and repository setup.
- Milestone 2: data acquisition, provenance audit, reproducible cleaning, quality assessment, and exploratory analysis.
- Milestone 3: modeling and evaluation with time-respecting validation, baselines and candidate methods for every research question, error analysis and the routed table. This is the current milestone.
- Later milestones: association strength and the popularity split, replenishment and sequential analysis, temporal validation, and the final routed rule set.

## Data use and privacy

Do not commit credentials, raw data, or anything that could re-identify a member or household. IDs appear only in processed files on your own machine. Results are reported only at the aggregate rule level.
