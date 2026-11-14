"""Обучение, версии моделей, отчёты."""
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from .data import data_files, load_dataset, to_long
from .model import TARGETS, add_base_features, coefficients, cross_validate, fit_final, predict_post, target_error
from .planner import nomogram_tables

UNCERTAINTY_LEVEL = 0.8


def write_excel(path, sheets):
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
            ws = xw.sheets[name[:31]]
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 70)
            ws.freeze_panes = "A2"


def latest_model_path(cfg):
    return cfg.models_dir / "актуальная.joblib"


def load_bundle(cfg):
    path = latest_model_path(cfg)
    if not path.exists():
        raise FileNotFoundError("Модель ещё не обучена: выполните  python -m nomogram train")
    return joblib.load(path)


def eye_key(df):
    return df["id"].astype(str) + "|" + df["eye"].astype(str)


def _prospective(prev, d, late_months):
    """Проверка прошлой версии модели на глазах, которых она не видела при обучении."""
    new = ~eye_key(d).isin(prev["eye_keys"])
    late = new & (d["months"] >= late_months)
    if late.sum() == 0:
        return None
    pred = predict_post(prev, d[late])
    err = (d.loc[late, "post_M"] - pred["post_M"]).abs()
    naive = (d.loc[late, "post_M"] - (d.loc[late, "pre_M"] - d.loc[late, "plan_M"])).abs()
    return {"версия": prev["version"], "новых глаз": int(eye_key(d[late]).nunique()),
            "визитов": int(late.sum()), "MAE M": float(err.mean()),
            "MAE M без поправки": float(naive.mean()), "±0.50 D, %": float(100 * (err <= 0.5).mean())}


def train(cfg, log=print):
    wide, issues = load_dataset(cfg)
    long = to_long(wide, cfg)
    used = wide[wide["excluded"] == ""]
    log(f"Глаз в таблицах: {len(wide)}, в обучении: {long['id'].str.cat(long['eye']).nunique()}, "
        f"визитов с рефракцией: {len(long)}")
    d, metrics, oof, best = cross_validate(long, cfg)
    models = fit_final(d, cfg, best)

    late_mo = cfg["model"]["late_months"]
    late = (d["months"] >= late_mo).values
    resid_m = target_error(d, "M").values - oof["M"][best["M"]]
    unc = {"level": UNCERTAINTY_LEVEL, "M": float(np.quantile(np.abs(resid_m[late]), UNCERTAINTY_LEVEL))}

    num_cols = ["age", "pre_cdva", "k1", "k1_axis", "k2", "k2_axis", "al", "acd", "pachy", "pupil", "oz", "flap"]
    typical = {c: float(used[c].median()) for c in num_cols}
    typical["oz"] = float(used["oz"].mode().iloc[0])
    eyes = d.drop_duplicates(["id", "eye"])
    pre_m = used["pre_sph"] + used["pre_cyl"] / 2
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle = {
        "version": version,
        "models": models,
        "model_kinds": best,
        "features": {t: list(cfg["model"][f"features_{t}"]) for t in TARGETS},
        "uncertainty": unc,
        "typical": typical,
        "ranges": {"pre_M": (float(pre_m.min()), float(pre_m.max())),
                   "cyl_abs": float(used["pre_cyl"].abs().max()),
                   "age": (float(used["age"].min()), float(used["age"].max()))},
        "n_hyperopes": int((eyes["plan_M"] > 0).sum()),
        "protocols": sorted(used["protocol"].dropna().astype(str).unique()),
        "protocol_counts": {str(k): int(v) for k, v in eyes["protocol"].dropna().astype(str).value_counts().items()},
        "eye_keys": set(eye_key(eyes)),
        "n_eyes": int(len(eyes)), "n_rows": int(len(d)),
        "ucva_cap": float(d["post_ucva"].max()),
    }

    prospective = None
    prev_path = latest_model_path(cfg)
    if prev_path.exists():
        try:
            prospective = _prospective(joblib.load(prev_path), d, late_mo)
        except Exception as e:  # несовместимая старая версия не должна мешать обучению
            log(f"Проверка прошлой модели пропущена: {e}")

    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    m_late = metrics[(metrics["компонент"] == "M")].set_index("модель")["MAE, от срока"]
    hist = {"версия": version, "глаз": bundle["n_eyes"], "визитов": bundle["n_rows"],
            "модель M": best["M"], "CV MAE M от срока": round(m_late[best["M"]], 3),
            "MAE M без поправки": round(m_late["без поправки (план = результат)"], 3),
            f"{UNCERTAINTY_LEVEL:.0%} глаз в пределах ± D": round(unc["M"], 2)}
    if prospective:
        hist.update({f"проспективно: {k}": v for k, v in prospective.items()})
    hist_path = cfg.models_dir / "история_обучения.csv"
    pd.concat([pd.read_csv(hist_path, encoding="utf-8-sig") if hist_path.exists() else pd.DataFrame(),
               pd.DataFrame([hist])]).to_csv(hist_path, index=False, encoding="utf-8-sig")

    summary = [
        ("Версия модели", version),
        ("Файлы данных", ", ".join(p.split("\\")[-1].split("/")[-1] for p in data_files(cfg))),
        ("Глаз в таблицах", len(wide)),
        ("Исключено глаз", int((wide["excluded"] != "").sum())),
        ("Глаз в обучении", bundle["n_eyes"]),
        ("Пациентов", int(d["id"].nunique())),
        ("Визитов с рефракцией", bundle["n_rows"]),
    ]
    summary += [(f"  визит '{v}'", int(n)) for v, n in d["visit"].value_counts(sort=False).items()]
    summary += [(f"Модель для {t}", best[t]) for t in TARGETS]
    summary += [
        (f"MAE сф. экв. (визиты ≥ {late_mo} мес), модель", round(m_late[best["M"]], 3)),
        ("MAE сф. экв. без поправки (как вводили)", round(m_late["без поправки (план = результат)"], 3)),
        (f"{UNCERTAINTY_LEVEL:.0%} глаз в пределах ± D (сф. экв.)", round(unc["M"], 2)),
        ("Гиперметропов в обучении", bundle["n_hyperopes"]),
        ("Глаз по протоколам", ", ".join(f"{k}: {v}" for k, v in sorted(bundle["protocol_counts"].items()))),
        ("Срок оптимизации номограммы, мес", cfg["planning"]["horizon_months"]),
        ("Ранний осмотр 'после операции' принят за, мес", cfg["visits"].get("после операции")),
        ("Замечаний к данным", len(issues)),
    ]
    if prospective:
        summary += [(f"Проверка прошлой модели на новых глазах: {k}", v) for k, v in prospective.items()]
    baseline = min(m_late["без поправки (план = результат)"], m_late["средняя поправка"])
    if m_late[best["M"]] > baseline:
        summary.append(("ВНИМАНИЕ", "модель не точнее простой поправки — данных мало или они шумные; "
                                    "используйте номограмму с осторожностью"))

    bundle["summary"] = summary
    bundle["metrics"] = metrics.round(3)
    bundle["trained_at"] = datetime.now().strftime("%d.%m.%Y %H:%M")
    joblib.dump(bundle, cfg.models_dir / f"модель_{version}.joblib")
    joblib.dump(bundle, prev_path)

    cv_pred = d[["_file", "_row", "id", "eye", "visit", "months", "pre_M", "plan_M", "post_M", "post_ucva"]].copy()
    cv_pred["прогноз M (CV)"] = (d["pre_M"] - d["plan_M"] + oof["M"][best["M"]]).round(2)
    cv_pred["ошибка прогноза"] = (cv_pred["post_M"] - cv_pred["прогноз M (CV)"]).round(2)
    cv_pred["ошибка без поправки"] = (cv_pred["post_M"] - (d["pre_M"] - d["plan_M"])).round(2)

    stamp = f"_{version}"
    write_excel(cfg.results_dir / f"отчёт_обучения{stamp}.xlsx", {
        "Сводка": pd.DataFrame(summary, columns=["показатель", "значение"]),
        "Метрики CV": metrics.round(3),
        "Коэффициенты": coefficients(models, cfg).round(4),
        "Качество данных": issues.frame(),
        "Прогноз по глазам (CV)": cv_pred,
    })
    write_tables(bundle, cfg)
    return bundle, summary, issues


def write_tables(bundle, cfg):
    sheets = {}
    for protocol in bundle["protocols"]:
        sph, cyl = nomogram_tables(bundle, cfg, protocol)
        sheets[f"{protocol} сфера"] = sph
        sheets[f"{protocol} цилиндр"] = cyl
    t = bundle["typical"]
    note = pd.DataFrame([
        ("Модель", bundle["version"]),
        ("Что в таблице", "Рекомендуемый ввод в лазер EX500 для субъективной рефракции"),
        ("Цель", f"сф. экв. {cfg['planning']['target_M']:+.2f} D через {cfg['planning']['horizon_months']} мес"),
        ("Астигматизм", "в таблице роговичный астигматизм принят равным рефракционному и на той же оси; "
                        "у конкретного пациента цилиндр и ось подбираются по его кератометрии"),
        ("Типичный глаз", f"Kср {(t['k1'] + t['k2']) / 2:.2f}, пахиметрия {t['pachy']:.0f}, "
                          f"лоскут {t['flap']:.0f}, AL {t['al']:.2f}, ОЗ {t['oz']}"),
        ("Важно", "Для конкретного пациента используйте команду plan: учитывается его биометрия"),
    ], columns=["параметр", "значение"])
    write_excel(cfg.results_dir / "номограмма_таблица.xlsx", {"Описание": note, **sheets})
