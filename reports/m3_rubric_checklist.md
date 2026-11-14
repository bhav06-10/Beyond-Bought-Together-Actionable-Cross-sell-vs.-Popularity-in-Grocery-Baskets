# Milestone 3 rubric checklist

Every item in the Milestone 3 brief, and where the evidence lives. Paths are relative to the repository root; the website section is "Models and Evaluation".

| Rubric item | Met by | Evidence |
| --- | --- | --- |
| 1. Modeling plan: question per method | Each RQ has its own script and question | Website modeling-plan table; `README.md` table; docstring at the top of every `src/m3/step*.py` |
| 1. Task type stated | Ranking, regression/calibration, classification, time-to-event, clustering, observational causal check | Same tables |
| 1. Why the methods fit the data | Sparse binary co-occurrence, thin pair counts, censored gaps, non-negative baskets | Website "Why these methods fit the data" |
| 1. Assumptions and failure modes of our implementation | Swap baseline keeps basket sizes and popularity; promotions are not randomized; one test quarter includes holidays | Website "What this does not prove"; `step11_promotions.py` docstring |
| 2. Baseline plus at least two candidates | RQ2: 2 baselines, 3 candidates. RQ1, RQ3, RQ4, RQ5: one baseline and at least two candidates each | `reports/tables/m3_rq2_test.csv`, `m3_rq1_measures.csv`, `m3_rq3_survival.csv`, `m3_rq4_next_trip.csv`, `m3_rq5_reliability.csv` |
| 3. Features used and excluded | 10 feature groups used; demographics, coupon discounts, sales value, product IDs excluded with reasons | `reports/tables/m3_features.csv`, `step02_features.py` |
| 3. Encoding, scaling, transformation | Log ratios, SVD of positive PMI, standardization inside the logistic pipelines | `features.py`, `step10`, `step12` |
| 3. No target or evaluation leakage | Every statistic fitted on the window before the scored period; hidden category never feeds its own features; labels from later data only for RQ1/RQ5 | `features.py`, `harness.py`, `step03_harness.py` |
| 3. Time-ordered split | Fit Jan-Jun, tune Jul-Sep, test Oct-Dec once (refit Jan-Sep); RQ5 Q1->Q2 train, Q3->Q4 test | `config.yaml` (`m3.periods`) |
| 4. Metrics that match the task and its cost | Cross-sell Recall@10 under a protocol where popularity is the signal-free reference; PR-AUC for the 24% failing rules; concordance and Brier for censored timing; stability for clusters | `harness.py`, website "Evaluation design" |
| 4. Association metrics: support, confidence, lift, redundancy control, actionable or obvious | Rules mined with support/confidence thresholds and lift above 1; minimum-improvement redundancy test (2.0% of two-item rules redundant, recall unchanged when removed); mirror pairs counted; a third of rules end in a staple, handled by the cross-sell protocol; the routed table separates actionable pairs from popularity co-occurrence | `step17_rule_redundancy.py`, `m3_rq2_rule_redundancy.csv`, `m3_rq2_rules_annotated.csv`, `step16_routing.py` |
| 4. Defensible validation | Time split, paired household bootstrap with Holm correction, arules 5-fold CV, Kaggle placebo, store transfer, outside replication on Instacart and a known-complement check on Carbo-Loading | `step07_final_test.py`, `step08_crosscheck.py`, `step15_external.py`, `m3_rq2_pairwise.csv`, `m3_ext_instacart_rq2.csv`, `m3_rq1_carbo_known_complements.csv`, figure 18 |
| 5. Only meaningful hyperparameters, documented search | C1 support/confidence grid; C2 measure x prior weight x clipping; C3 10-trial random search with early stopping; RQ3/RQ5 small grids | `config.yaml`, `m3_rq2_tuning_c1_c2.csv`, `m3_rq2_tuning_c3.csv`, `m3_rq3_tuning.csv` |
| 5. No tuning on the test set | All tuning on Jul-Sep (or grouped CV inside the training quarter for RQ5) | Same |
| 5. Same protocol for every model | All RQ2 models score the same 263,004 queries | `step07_final_test.py` |
| 5. Prefer the simpler model when close | Logistic over LightGBM for RQ5; median gap kept for reminder timing; C2 is the explainable fallback to C3 | Website RQ3 and RQ5 captions |
| 6. What the model found, in terms of the question | The routed table: 1,835 bundles, 73 reminder categories, the rest left alone | `m3_routed_pairs.csv`, `m3_routed_categories.csv`, figure 17 |
| 6. Errors, difficult cases, failure modes | Recall by target type, basket size, history, department; substitutes; exemplar wins and misses | `m3_err_*.csv`, figure 16 |
| 6. Interpretation tools used correctly | SHAP for C3 (read as reliance), coefficients for logistic models, mission profiles for NMF | `m3_err_c3_shap.csv`, `m3_rq5_logistic_coefficients.csv`, `m3_rq4_mission_profiles.csv` |
| 6. Subgroup performance | Income, household size, kids for the 801 households with demographics | `m3_err_subgroups.csv` |
| 6. What the evaluation does not prove | Offline hits are not sales lift; promotions check is observational and underpowered | Website "What this does not prove" |
| Website | Models and Evaluation section with plan, design, results, comparison tables, figures 9 to 18 (including outside validation), and a link to the code for every RQ | `docs/index.md` |
| Repository | One script per step, `make m3`, seeds and grids in `config.yaml`, generated tables and figures | `src/m3/`, `Makefile`, `reports/`, `figures/` |
