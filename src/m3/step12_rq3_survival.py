"""Step 12. RQ3: when will a household rebuy a category?  Task type: time-to-event forecasting
(discrete-time survival).

Unit      a household and category with at least 4 purchase days before the cutoff
Cutoffs   train/tune: history to 30 June, outcome July to September
          test:       history to 30 September, outcome October to December
Outcome   days from the cutoff to the next purchase (1 to 91), censored at 91 (end of panel)
Baseline  the household's own median gap: next purchase = last purchase + median gap
Candidates  discrete-time hazard logistic regression; gradient-boosted discrete-time hazard (LightGBM)
Novel feature  relative load: units x parsed package size at the last purchase, divided by the
          household's usual load for that category (a big pack should push the next trip back)
Metrics   concordance index (ranking who rebuys first), integrated Brier score over 91 days, and
          for reminders: share of purchases where the predicted day is within 7 days of the real one
"""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from m3lib import OUT, TAB, FIG, SEED, M3, T1, T2, T3, m3_numbers, plt, BLUE, ORANGE, GREY, MUTED
from features import load

H = M3["rq3"]["horizon_days"]; MINP = M3["rq3"]["min_purchases"]
rng = np.random.default_rng(SEED)
L, items = load()
loads = pd.read_csv(OUT / "cj_loads.csv.gz", dtype={"basket": str})
L = L.merge(loads, on=["basket", "label"], how="left")
D = (L.groupby(["member", "label", "date"]).agg(load=("load", "sum"), units=("units", "sum")).reset_index()
     .sort_values(["member", "label", "date"]))
D.loc[D.load <= 0, "load"] = np.nan


def snapshot(cut, end):
    past = D[D.date < cut].copy()
    past["gap"] = past.groupby(["member", "label"]).date.diff().dt.days
    g = past.groupby(["member", "label"])
    f = g.agg(n=("date", "size"), last=("date", "max"), first=("date", "min"), gap_med=("gap", "median"), gap_mean=("gap", "mean"),
              gap_sd=("gap", "std"), last_gap=("gap", "last"), last_load=("load", "last"), med_load=("load", "median"),
              last_units=("units", "last"), med_units=("units", "median"))
    f = f[f.n >= MINP].copy()
    f["elapsed"] = (cut - f["last"]).dt.days
    f["cv"] = f.gap_sd / f.gap_mean
    f["rel_load"] = (f.last_load / f.med_load).fillna(f.last_units / f.med_units).clip(0.1, 10)
    f["rel_units"] = (f.last_units / f.med_units).clip(0.1, 10)
    f["rate30"] = f.n / ((cut - f["first"]).dt.days.clip(lower=1) / 30)
    trips = L[L.date < cut].drop_duplicates(["member", "date"]).groupby("member").date.agg(["size", "min"])
    f["hh_trips_week"] = (trips["size"] / ((cut - trips["min"]).dt.days.clip(lower=7) / 7)).reindex(f.index.get_level_values(0)).values
    catgap = past.groupby("label").gap.median()
    f["cat_gap"] = catgap.reindex(f.index.get_level_values(1)).values
    f["clock0"] = f.elapsed / f.gap_med
    fut = D[(D.date >= cut) & (D.date < end)].groupby(["member", "label"]).date.min()
    dd = (fut.reindex(f.index) - cut).dt.days + 1
    f["event"] = dd.notna() & (dd <= H)
    f["duration"] = dd.where(f.event, H).astype(int)
    return f.reset_index()


STATIC = ["elapsed", "gap_med", "gap_mean", "cv", "last_gap", "n", "rel_load", "rel_units", "rate30", "hh_trips_week", "cat_gap"]


def person_period(f, days=None):
    """One row per pair and day at risk (1..duration); event on the last day if it happened."""
    dur = f.duration.values if days is None else np.full(len(f), days)
    rep = np.repeat(np.arange(len(f)), dur)
    t = np.concatenate([np.arange(1, d + 1) for d in dur])
    X = f[STATIC].iloc[rep].reset_index(drop=True).fillna(f[STATIC].median())
    X["t"] = t; X["log_t"] = np.log(t)
    X["clock"] = (X.elapsed + t) / X.gap_med.clip(lower=1)
    X["clock_cat"] = (X.elapsed + t) / X.cat_gap.clip(lower=1)
    for k in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]:          # piecewise terms so the logistic hazard can bend
        X[f"clock_gt_{k}"] = np.maximum(X.clock - k, 0)
    X["log_rel_load"] = np.log(X.rel_load)
    y = np.zeros(len(t), int)
    if days is None:
        last = np.cumsum(dur) - 1; y[last[f.event.values]] = 1
    return X, y, rep


def survival(model, f):
    X, _, rep = person_period(f, H)
    h = np.clip(model.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6).reshape(len(f), H)
    return np.cumprod(1 - h, axis=1)


def cindex(time, event, risk, n=8000):
    i = rng.choice(len(time), min(n, len(time)), replace=False); t, e, r = time[i], event[i], risk[i]
    conc = ties = tot = 0
    order = np.argsort(t)
    for a in order[e[order]]:
        later = t > t[a]
        tot += later.sum(); conc += (r[a] > r[later]).sum(); ties += (r[a] == r[later]).sum()
    return (conc + 0.5 * ties) / tot


def evaluate(name, f, S=None, pred_day=None):
    tt = np.arange(1, H + 1)
    if S is None:                                           # baseline: step survival at the predicted day
        S = (tt[None, :] < pred_day[:, None]).astype(float)
    if pred_day is None:
        pred_day = np.where((S <= 0.5).any(1), (S <= 0.5).argmax(1) + 1, H + 1)
    rmst = S.sum(1)
    obs = (f.duration.values[:, None] > tt[None, :]).astype(float)
    ibs = float(((obs - S) ** 2).mean())
    ev = f.event.values
    err = np.abs(pred_day[ev] - f.duration.values[ev])
    return dict(model=name, c_index=cindex(f.duration.values, ev, -rmst), integrated_brier=ibs,
                reminder_within_7d=float((err <= 7).mean()), mae_days_events=float(err.mean()),
                pairs=int(len(f)), event_rate=float(ev.mean()))


tr = snapshot(T1, T2); te = snapshot(T2, T3)
print("train pairs", len(tr), "event rate", tr.event.mean(), "| test pairs", len(te), "event rate", te.event.mean(), flush=True)
# tuning split by household
mem = tr.member.unique(); val = set(rng.choice(mem, len(mem) // 5, replace=False))
fit_f = tr[~tr.member.isin(val)].sample(min(20000, (~tr.member.isin(val)).sum()), random_state=SEED)
val_f = tr[tr.member.isin(val)].sample(min(5000, tr.member.isin(val).sum()), random_state=SEED)
Xf, yf, _ = person_period(fit_f)
print("person-period rows", len(yf), flush=True)
cands = {}
for C in [0.1, 1.0]:
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(Xf, yf)
    cands[("logit", C)] = (m, evaluate("", val_f, survival(m, val_f))["integrated_brier"])
for nl in [15, 31]:
    for lr in [0.05, 0.1]:
        m = lgb.LGBMClassifier(n_estimators=300, num_leaves=nl, learning_rate=lr, min_child_samples=100, random_state=SEED, verbose=-1, n_jobs=2).fit(Xf, yf)
        cands[("lgb", nl, lr)] = (m, evaluate("", val_f, survival(m, val_f))["integrated_brier"])
tun = pd.DataFrame([dict(model=k[0], params=str(k[1:]), val_integrated_brier=v[1]) for k, v in cands.items()])
tun.to_csv(TAB / "m3_rq3_tuning.csv", index=False); print(tun)
best_lr = min((k for k in cands if k[0] == "logit"), key=lambda k: cands[k][1])
best_gb = min((k for k in cands if k[0] == "lgb"), key=lambda k: cands[k][1])
# refit on all training pairs (subsampled for memory) with the chosen settings, then test once
all_f = tr.sample(min(30000, len(tr)), random_state=SEED); Xa, ya, _ = person_period(all_f)
m_lr = make_pipeline(StandardScaler(), LogisticRegression(C=best_lr[1], max_iter=3000)).fit(Xa, ya)
m_gb = lgb.LGBMClassifier(n_estimators=300, num_leaves=best_gb[1], learning_rate=best_gb[2], min_child_samples=100,
                          random_state=SEED, verbose=-1, n_jobs=2).fit(Xa, ya)
base_day = (te.gap_med.fillna(te.cat_gap) - te.elapsed).clip(lower=0).round().astype(int).values + 1
rows = [evaluate("Personal median gap (baseline)", te, pred_day=base_day)]
S_lr = survival(m_lr, te); S_gb = survival(m_gb, te)
rows += [evaluate("Discrete-time hazard, logistic", te, S_lr), evaluate("Discrete-time hazard, LightGBM", te, S_gb)]
R = pd.DataFrame(rows); R.to_csv(TAB / "m3_rq3_survival.csv", index=False); print(R.round(4))
names = list(Xa.columns)
coef = pd.Series(m_lr[-1].coef_[0], index=names); coef.rename("standardized_coef").to_csv(TAB / "m3_rq3_logistic_coefficients.csv")
te_out = te[["member", "label", "elapsed", "gap_med", "cv", "rel_load", "event", "duration"]].copy()
te_out["p_buy_7d_lgb"] = 1 - S_gb[:, 6]; te_out["pred_day_lgb"] = np.where((S_gb <= 0.5).any(1), (S_gb <= 0.5).argmax(1) + 1, H + 1)
te_out.to_csv(OUT / "rq3_test_predictions.csv.gz", index=False)

# relative load check on the test quarter (raw data, no model): bigger last load -> later rebuy?
q = pd.qcut(te.rel_load.rank(method="first"), 5, labels=["smallest", "small", "usual", "large", "largest"])
load_eff = te.assign(q=q).groupby("q", observed=True).apply(lambda d: pd.Series(dict(
    median_days=d.duration[d.event].median(), rebuy_rate=d.event.mean(), pairs=len(d))))
load_eff.to_csv(TAB / "m3_rq3_relative_load.csv"); print(load_eff)

fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.4))
cols = [GREY, BLUE, ORANGE]
for ax, met, lab in [(axs[0], "c_index", "Concordance (higher is better)"), (axs[1], "reminder_within_7d", "Reminders within 7 days of the purchase")]:
    ax.bar(range(3), R[met], color=cols)
    for i, v in enumerate(R[met]): ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=11)
    ax.set_xticks(range(3)); ax.set_xticklabels(["Median gap", "Hazard\nlogistic", "Hazard\nLightGBM"]); ax.set(ylim=(0, 1), title=lab)
axs[2].plot(range(5), load_eff.median_days, marker="o", color=ORANGE, lw=2)
axs[2].set_xticks(range(5)); axs[2].set_xticklabels(load_eff.index)
axs[2].set(xlabel="Last purchase size vs the household's usual (quintile)", ylabel="Median days to rebuy (test quarter)",
           ylim=(0, 30), title="Pack size does not change time to rebuy")
fig.suptitle("Figure 14. RQ3: forecasting when a household rebuys a category (test: October to December)", x=0.01, ha="left", fontsize=14.5, y=1.03)
fig.tight_layout(); fig.savefig(FIG / "fig14_rq3_survival.png"); plt.close(fig)
m3_numbers("rq3", dict(train_pairs=int(len(tr)), test_pairs=int(len(te)), test_event_rate=float(te.event.mean()),
                       best_logistic_C=best_lr[1], best_lgb=dict(num_leaves=best_gb[1], learning_rate=best_gb[2]),
                       results=R.set_index("model").to_dict("index"),
                       relative_load=load_eff.to_dict("index"),
                       logistic_coef_log_rel_load=float(coef["log_rel_load"])))
