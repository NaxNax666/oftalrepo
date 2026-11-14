"""Прямая модель: предсказание послеоперационной рефракции по данным до операции и плану лазера.

Для каждого компонента вектора силы (M, J0, J45) модель предсказывает отклонение
фактического результата от "идеального": ошибка = послеоп. − (до операции − план).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, LinearRegression, RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import optics

TARGETS = ["M", "J0", "J45"]


def add_base_features(df):
    """Вычисляемые признаки. Нужны колонки до операции, плана и срок `months`."""
    d = df.copy()
    for p in ["pre", "plan"]:
        m, j0, j45 = optics.to_vectors(d[f"{p}_sph"], d[f"{p}_cyl"].fillna(0), d[f"{p}_axis"].fillna(0))
        d[f"{p}_M"], d[f"{p}_J0"], d[f"{p}_J45"] = m, j0, j45
    d["plan_Cabs"] = d["plan_cyl"].abs()
    d["k_mean"] = (d["k1"] + d["k2"]) / 2
    d["corneal_J0"], d["corneal_J45"] = optics.corneal_vectors(d["k1"], d["k1_axis"], d["k2"], d["k2_axis"])
    d["hyper"] = (d["plan_M"] > 0).astype(float)
    d["log_t"] = np.log1p(d["months"].astype(float) * 30.4)
    d["pre_cdva_logmar"] = to_logmar(d["pre_cdva"]) if "pre_cdva" in d else np.nan
    # Внутренний (нероговичный) астигматизм: рефракционный минус роговичный
    d["ora_J0"], d["ora_J45"] = d["pre_J0"] - d["corneal_J0"], d["pre_J45"] - d["corneal_J45"]
    # Рефракция InnovEyes (протокол WLP): отличие от субъективной; 0, если не измерялась
    if "ie_sph" in d:
        num = {c: pd.to_numeric(d[c], errors="coerce") if c in d else np.nan for c in ("ie_sph", "ie_cyl", "ie_axis")}
        m, j0, j45 = optics.to_vectors(num["ie_sph"], pd.Series(num["ie_cyl"], index=d.index).fillna(0),
                                       pd.Series(num["ie_axis"], index=d.index).fillna(0))
        d["ie_dM"] = pd.Series(m - d["pre_M"], index=d.index).fillna(0.0)
        d["ie_dJ0"] = pd.Series(j0 - d["pre_J0"], index=d.index).fillna(0.0)
        d["ie_dJ45"] = pd.Series(j45 - d["pre_J45"], index=d.index).fillna(0.0)
    else:
        d["ie_dM"] = d["ie_dJ0"] = d["ie_dJ45"] = 0.0
    return d


def to_logmar(decimal):
    v = pd.to_numeric(pd.Series(decimal), errors="coerce").astype(float).values
    return -np.log10(np.where(v > 0, v, np.nan))


def to_decimal(logmar):
    return 10 ** (-np.asarray(logmar, dtype=float))


UCVA_FEATURES = ["pre_cdva_logmar", "res_myopia", "res_hyperopia", "res_cyl", "early"]


def ucva_design(d, post_m, post_j0, post_j45):
    """Признаки остроты без коррекции: острота с коррекцией до операции + остаточная рефракция."""
    post_m = np.asarray(post_m, dtype=float)
    return pd.DataFrame({
        "pre_cdva_logmar": np.asarray(d["pre_cdva_logmar"], dtype=float),
        "res_myopia": np.maximum(-post_m, 0),
        "res_hyperopia": np.maximum(post_m, 0),
        "res_cyl": 2 * np.hypot(np.asarray(post_j0, dtype=float), np.asarray(post_j45, dtype=float)),
        "early": (np.asarray(d["months"], dtype=float) < 0.5).astype(float),
    }, index=d.index)


def make_ucva_estimator():
    # Все коэффициенты ≥ 0: хуже острота с коррекцией / больше остаточная ошибка → хуже UCVA
    return make_pipeline(SimpleImputer(strategy="median"), LinearRegression(positive=True))


def cross_validate_ucva(d, folds, late):
    ok = d["post_ucva"].notna().values & (d["post_ucva"] > 0).values
    y = to_logmar(d["post_ucva"])
    X = ucva_design(d, d["post_M"], d["post_J0"], d["post_J45"])
    pred = {"средняя": np.full_like(y, np.nan), "линейная (коэф. ≥ 0)": np.full_like(y, np.nan)}
    for tr, te in folds:
        tr, te = tr[ok[tr]], te[ok[te]]
        pred["средняя"][te] = np.mean(y[tr])
        pred["линейная (коэф. ≥ 0)"][te] = make_ucva_estimator().fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
    rows = []
    for name, p in pred.items():
        r = np.abs(y - p)[ok & late]
        rows.append({"компонент": "UCVA, logMAR", "модель": name, "MAE, все визиты": np.nanmean(np.abs(y - p)[ok]),
                     "MAE, от срока": np.mean(r), "±1 строка, от срока, %": 100 * np.mean(r <= 0.1)})
    return pd.DataFrame(rows)


def fit_ucva(d):
    ok = d["post_ucva"].notna() & (d["post_ucva"] > 0)
    X = ucva_design(d[ok], d.loc[ok, "post_M"], d.loc[ok, "post_J0"], d.loc[ok, "post_J45"])
    return make_ucva_estimator().fit(X, to_logmar(d.loc[ok, "post_ucva"]))


def design(df, features):
    cols = {}
    for f in features:
        if "*" in f:
            a, b = f.split("*")
            cols[f] = pd.to_numeric(df[a], errors="coerce") * pd.to_numeric(df[b], errors="coerce")
        elif "=" in f:
            a, b = f.split("=")
            cols[f] = (df[a].astype(str).str.lower() == b.lower()).astype(float)
        else:
            cols[f] = pd.to_numeric(df[f], errors="coerce")
    return pd.DataFrame(cols, index=df.index)


def make_estimator(kind):
    if kind == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             RidgeCV(alphas=np.logspace(-2, 3, 30)))
    if kind == "huber":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=2000))
    if kind == "gbm":
        return HistGradientBoostingRegressor(loss="absolute_error", max_depth=3, learning_rate=0.05,
                                             max_iter=300, min_samples_leaf=20, random_state=0)
    raise ValueError(f"Неизвестный тип модели: {kind}")


def target_error(d, t):
    return d[f"post_{t}"] - (d[f"pre_{t}"] - d[f"plan_{t}"])


def _metrics(err_true, err_pred, late):
    r = err_true - err_pred
    out = {"MAE, все визиты": np.mean(np.abs(r)), "MAE, от срока": np.mean(np.abs(r[late]))}
    out["±0.50 D, от срока, %"] = 100 * np.mean(np.abs(r[late]) <= 0.5)
    out["±1.00 D, от срока, %"] = 100 * np.mean(np.abs(r[late]) <= 1.0)
    return out


def cross_validate(long, cfg):
    """Групповая кросс-валидация (оба глаза пациента всегда в одной части)."""
    d = add_base_features(long)
    groups = d["id"].values
    n_splits = min(cfg["model"]["cv_folds"], len(np.unique(groups)))
    folds = list(GroupKFold(n_splits=n_splits).split(d, groups=groups))
    late = (d["months"] >= cfg["model"]["late_months"]).values
    rows, oof = [], {}
    for t in TARGETS:
        y = target_error(d, t).values
        X = design(d, cfg["model"][f"features_{t}"])
        preds = {"без поправки (план = результат)": np.zeros_like(y), "средняя поправка": np.zeros_like(y)}
        for kind in cfg["model"]["candidates"]:
            preds[kind] = np.zeros_like(y)
        for tr, te in folds:
            preds["средняя поправка"][te] = np.mean(y[tr])
            for kind in cfg["model"]["candidates"]:
                est = make_estimator(kind).fit(X.iloc[tr], y[tr])
                preds[kind][te] = est.predict(X.iloc[te])
        for name, p in preds.items():
            rows.append({"компонент": t, "модель": name, **_metrics(y, p, late)})
        oof[t] = preds
    metrics = pd.concat([pd.DataFrame(rows), cross_validate_ucva(d, folds, late)], ignore_index=True)
    best = {}
    for t in TARGETS:
        m = metrics[(metrics["компонент"] == t) & metrics["модель"].isin(cfg["model"]["candidates"])]
        best[t] = m.sort_values("MAE, от срока").iloc[0]["модель"]
    return d, metrics, oof, best


def fit_final(d, cfg, best):
    models = {}
    for t in TARGETS:
        y = target_error(d, t).values
        X = design(d, cfg["model"][f"features_{t}"])
        models[t] = make_estimator(best[t]).fit(X, y)
    models["UCVA"] = fit_ucva(d)
    return models


def coefficients(models, cfg):
    """Коэффициенты линейных моделей в исходных единицах (изменение ошибки, D, на 1 единицу признака)."""
    rows = []
    for t, est in models.items():
        if t == "UCVA":
            reg = est[-1]
            rows.append({"компонент": "UCVA, logMAR", "признак": "(свободный член)", "коэффициент": reg.intercept_})
            rows += [{"компонент": "UCVA, logMAR", "признак": f, "коэффициент": c}
                     for f, c in zip(UCVA_FEATURES, reg.coef_)]
            continue
        if not hasattr(est, "named_steps"):
            continue
        scaler = est.named_steps["standardscaler"]
        reg = est[-1]
        coef = reg.coef_ / scaler.scale_
        intercept = reg.intercept_ - np.sum(coef * scaler.mean_)
        rows.append({"компонент": t, "признак": "(свободный член)", "коэффициент": intercept})
        for f, c, cs in zip(cfg["model"][f"features_{t}"], coef, reg.coef_):
            rows.append({"компонент": t, "признак": f, "коэффициент": c, "станд. коэффициент": cs})
    return pd.DataFrame(rows)


def predict_post(bundle, df):
    """Прогноз послеоперационных M, J0, J45 для таблицы с данными до операции, планом и сроком."""
    d = add_base_features(df)
    out = pd.DataFrame(index=d.index)
    for t in TARGETS:
        X = design(d, bundle["features"][t])
        out[f"post_{t}"] = d[f"pre_{t}"] - d[f"plan_{t}"] + bundle["models"][t].predict(X)
    if "UCVA" in bundle["models"]:
        X = ucva_design(d, out["post_M"], out["post_J0"], out["post_J45"])
        out["post_ucva_logmar"] = bundle["models"]["UCVA"].predict(X)
    return out
