# Run from the repository root.  `make all` rebuilds every table and figure from the raw files.
PY := python
M3 := cd src/m3 && $(PY)
.PHONY: all data clean_data audit pairs temporal m2 m3 m3_data m3_rq2 m3_rq1 m3_rq5 m3_promo m3_rq3 m3_rq4 m3_errors m3_external m3_routing tidy

all: m2 m3
m2: data clean_data audit pairs temporal

data:        ## download sources (Kaggle by hand, see data/README.md) and check hashes
	cd src && $(PY) acquire.py
clean_data:  ## steps 2-6: parse, standardise labels, build line tables
	cd src && $(PY) clean.py
audit:       ## provenance audit of the Kaggle file (figures 1-2)
	cd src && $(PY) audit.py
pairs:       ## same-basket structure, RQ1-RQ2 (figures 3-5)
	cd src && $(PY) pairs.py
temporal:    ## repurchase, next visit, stability, RQ3-RQ5 (figures 6-8)
	cd src && $(PY) temporal.py

# ---------------- Milestone 3: modeling and evaluation (needs `make m2` first) ----------------
m3: m3_data m3_rq2 m3_rq1 m3_rq5 m3_promo m3_rq3 m3_rq4 m3_errors m3_external m3_routing

m3_data:     ## steps 1-3: companion tables, past-only features, query sets
	$(M3) step01_data.py
	$(M3) step02_features.py
	$(M3) step03_harness.py
m3_rq2:      ## steps 4-8 and 17: baselines, tuning C1/C2, C3 learning-to-rank, final test, cross-check + placebo, rule redundancy
	$(M3) step04_baselines.py
	$(M3) step05_candidates.py
	$(M3) step06_ltr.py
	$(M3) step07_final_test.py
	$(M3) step08_crosscheck.py
	$(M3) step17_rule_redundancy.py
m3_rq1:      ## step 9: which pair measure predicts next-half strength (+ Carbo-Loading if present)
	$(M3) step09_rq1_measures.py
m3_rq5:      ## step 10: which rules fail next quarter
	$(M3) step10_rq5_reliability.py
m3_promo:    ## step 11: promotions natural experiment
	$(M3) step11_promotions.py
m3_rq3:      ## step 12: repurchase timing (discrete-time survival)
	$(M3) step12_rq3_survival.py
m3_rq4:      ## step 13: sequence vs taste, shopping missions
	$(M3) step13_rq4_sequence.py
m3_errors:   ## step 14: errors, subgroups, coupons, SHAP, exemplars
	$(M3) step14_errors.py
m3_external: ## step 15: Instacart replication (skips cleanly if the files are absent)
	$(M3) step15_external.py
m3_routing:  ## step 16: the routed table
	$(M3) step16_routing.py

tidy:        ## remove generated outputs (raw downloads are kept)
	rm -rf data/processed figures/*.png reports/tables/*.csv reports/numbers.json
