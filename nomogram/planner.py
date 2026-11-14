"""Номограмма как обратная задача: подбор ввода в лазер, дающего целевую рефракцию на заданный срок."""
import numpy as np
import pandas as pd

from . import optics
from .model import predict_post, to_decimal

TRAJECTORY_MONTHS = {"1 сут": 0.033, "1 мес": 1, "3 мес": 3, "6 мес": 6, "12 мес": 12}


def _num(p, key, default=np.nan):
    v = p.get(key, default)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return default if np.isnan(v) else v


def _valid(v):
    return v is not None and not np.isnan(v)


def _axis_candidates(ref_axis, cor_axis, step):
    """Оси ввода: на кратчайшей дуге между осью рефракции и осью роговичного астигматизма."""
    if not _valid(cor_axis):
        return [ref_axis % 180]
    if not _valid(ref_axis):
        return [round(cor_axis / step) * step % 180]
    diff = optics.axis_diff(ref_axis, cor_axis)
    lo, hi = sorted([ref_axis, ref_axis + diff])
    axes = [ref_axis, ref_axis + diff]
    k = np.ceil(lo / step) * step
    while k <= hi + 1e-9:
        axes.append(k)
        k += step
    return sorted({round(a) % 180 for a in axes})


def corneal_info(p, pc):
    """Роговичный астигматизм глаза и то, можно ли опираться на его ось при подборе."""
    mag, flat = optics.corneal_astigmatism(p["k1"], p["k1_axis"], p["k2"], p["k2_axis"])
    info = {"cyl": mag, "flat_axis": flat, "usable": False, "note": None}
    if not _valid(mag):
        info["note"] = "нет кератометрии — роговичный астигматизм не учтён"
    elif not _valid(flat):
        info["note"] = "не указаны оси K1/K2 — ось роговичного астигматизма не учтена"
    elif _valid(p["k1_axis"]) and _valid(p["k2_axis"]) and \
            abs(abs(optics.axis_diff(p["k1_axis"], p["k2_axis"])) - 90) > pc.get("k_axes_tolerance", 20):
        info["note"] = (f"оси K1 ({p['k1_axis']:.0f}°) и K2 ({p['k2_axis']:.0f}°) не перпендикулярны — "
                        "проверьте кератометрию, ось роговицы при подборе не использована")
    elif mag >= pc.get("corneal_min_cyl", 0.25):
        info["usable"] = True
    return info


def _candidates(p, pc, cornea):
    """Сетка вариантов ввода (сфера, цилиндр, ось).

    Сфера и величина цилиндра — вокруг субъективной рефракции; ось — между осью субъективного
    и осью роговичного астигматизма. Знак цилиндра — как в субъективной рефракции.
    """
    step = pc["step"]
    s0, c0 = p["pre_sph"], p["pre_cyl"]
    sph = np.round((s0 + np.arange(-pc["sph_search"], pc["sph_search"] + 1e-9, step)) / step) * step
    sign = 1.0 if c0 > 0 else -1.0
    # цилиндр можно уменьшать на cyl_search, а увеличивать — не более чем на cyl_increase:
    # усиление цилиндра сверх субъективного в данных встречается редко, и прогноз там ненадёжен
    mags = abs(c0) + np.arange(-pc["cyl_search"], pc.get("cyl_increase", pc["cyl_search"]) + 1e-9, step)
    mags = np.unique(np.round(mags[mags >= -1e-9] / step) * step)
    # ось роговицы в тех же обозначениях, что и рефракция: «−» цилиндр — плоская, «+» — крутая
    cor_axis = (cornea["flat_axis"] + (90 if sign > 0 else 0)) % 180 if cornea["usable"] else np.nan
    ref_axis = p["pre_axis"] if c0 != 0 else np.nan
    if pc.get("use_corneal_astigmatism", True):
        axes = _axis_candidates(ref_axis, cor_axis, pc.get("axis_step", 5)) if (c0 != 0 or cornea["usable"]) else []
    else:
        axes = [ref_axis % 180] if c0 != 0 else []
    if not axes:
        mags, axes = np.array([0.0]), [0]
    rows = []
    for m in mags:
        for a in (axes if m > 0 else axes[:1]):
            rows += [(s, sign * m if m else 0.0, a if m else 0) for s in sph]
    out = np.array(rows, dtype=float)
    return out[:, 0], out[:, 1], out[:, 2]


def _warnings(p, bundle, pc):
    w = []
    rng = bundle["ranges"]
    pre_m = p["pre_sph"] + p["pre_cyl"] / 2
    if not rng["pre_M"][0] <= pre_m <= rng["pre_M"][1]:
        w.append(f"сф. экв. {pre_m:+.2f} D вне диапазона обучающих данных "
                 f"[{rng['pre_M'][0]:+.2f}; {rng['pre_M'][1]:+.2f}] — экстраполяция")
    if abs(p["pre_cyl"]) > rng["cyl_abs"]:
        w.append(f"цилиндр {p['pre_cyl']:+.2f} D больше максимального в данных ({rng['cyl_abs']:.2f})")
    if not np.isnan(p["age"]) and not rng["age"][0] <= p["age"] <= rng["age"][1]:
        w.append(f"возраст {p['age']:.0f} вне диапазона данных {rng['age']}")
    if pre_m > 0 and bundle["n_hyperopes"] < pc["min_hyperopes"]:
        w.append(f"гиперметропия: в обучении всего {bundle['n_hyperopes']} глаз — прогноз ненадёжен")
    proto = str(p.get("protocol"))
    n_proto = bundle.get("protocol_counts", {}).get(proto, None if proto in bundle["protocols"] else 0)
    if n_proto == 0:
        w.append(f"протокол {proto}: в обучении ещё нет глаз — расчёт по общей модели без поправки на протокол "
                 "(поправка появится по мере внесения результатов и переобучения)")
    elif n_proto is not None and n_proto < pc.get("min_protocol_eyes", 30):
        w.append(f"протокол {proto}: в обучении всего {n_proto} глаз — поправка на протокол ненадёжна")
    return w


def plan_eye(bundle, patient, cfg, horizon=None, target=None):
    pc = cfg["planning"]
    horizon = pc["horizon_months"] if horizon is None else horizon
    p = dict(patient)
    for k in ["age", "pre_sph", "pre_cyl", "pre_axis", "pre_cdva", "k1", "k1_axis", "k2", "k2_axis",
              "al", "acd", "pachy", "pupil", "oz", "flap", "ie_sph", "ie_cyl", "ie_axis"]:
        p[k] = _num(p, k)
    ie_only = np.isnan(p["pre_sph"]) and not np.isnan(p["ie_sph"])
    if ie_only:  # субъективной рефракции нет — расчёт от рефракции InnovEyes
        p["pre_sph"], p["pre_cyl"], p["pre_axis"] = p["ie_sph"], p["ie_cyl"], p["ie_axis"]
    p["pre_cyl"] = 0.0 if np.isnan(p["pre_cyl"]) else p["pre_cyl"]
    p["pre_axis"] = 0.0 if np.isnan(p["pre_axis"]) else p["pre_axis"]
    p["flap"] = pc["default_flap"] if np.isnan(p["flap"]) else p["flap"]
    p["oz"] = bundle["typical"]["oz"] if np.isnan(p["oz"]) else p["oz"]
    if target is None:
        target = _num(p, "target_M", pc["target_M"])

    cornea = corneal_info(p, pc)
    best_c = optics.corneal_vectors(p["k1"], p["k1_axis"], p["k2"], p["k2_axis"])
    sph, cyl, axis = _candidates(p, pc, cornea)
    cand = pd.DataFrame([p] * len(sph))
    cand["plan_sph"], cand["plan_cyl"], cand["plan_axis"] = sph, cyl, axis
    cand["months"] = horizon
    pred = predict_post(bundle, cand)
    # при равных прогнозах — вариант, ближайший к субъективной рефракции (в векторах силы)
    _, pj0, pj45 = optics.to_vectors(p["pre_sph"], p["pre_cyl"], p["pre_axis"])
    _, cj0, cj45 = optics.to_vectors(sph, cyl, axis)
    # Отход цилиндра/оси от субъективной рефракции «стоит» cyl_change_weight: ввод меняется только
    # тогда, когда прогнозный выигрыш в остаточном астигматизме заметно больше погрешности модели.
    cyl_shift = (cj0 - pj0) ** 2 + (cj45 - pj45) ** 2
    cyl_penalty = pc.get("cyl_change_weight", 0.1) * cyl_shift
    tie_sph = 1e-3 * (sph - p["pre_sph"]) ** 2
    by_ucva = pc.get("criterion", "se") == "ucva" and abs(target) < 1e-9 and "post_ucva_logmar" in pred
    if by_ucva:
        criterion = "максимальная острота без коррекции"
        obj = pred["post_ucva_logmar"] + pc["se_weight_ucva"] * (pred["post_M"] - target) ** 2
        i = int(np.argmin(obj.values + cyl_penalty + tie_sph + 1e-3 * cyl_shift))
    else:
        # Цель — указание программе. Подбор в два шага, чтобы цилиндр не использовался для подгонки
        # сфероэквивалента: 1) цилиндр и ось — минимум остаточного астигматизма;
        # 2) сфера — прогнозный сфероэквивалент на заданный срок как можно ближе к цели.
        criterion = f"сфероэквивалент {target:+.2f} D через {horizon:g} мес, минимум остаточного астигматизма"
        astig = (pred["post_J0"] ** 2 + pred["post_J45"] ** 2).values + cyl_penalty + 1e-3 * cyl_shift
        j = int(np.argmin(astig))
        same = np.flatnonzero((cyl == cyl[j]) & (axis == axis[j]))
        i = int(same[np.argmin(((pred["post_M"].values - target) ** 2 + tie_sph)[same])])
    best = cand.iloc[[i]].copy()
    bp = pred.iloc[i]

    manifest = cand.iloc[[0]].copy()
    manifest["plan_sph"], manifest["plan_cyl"], manifest["plan_axis"] = p["pre_sph"], p["pre_cyl"], p["pre_axis"]
    mp = predict_post(bundle, manifest).iloc[0]

    traj = pd.concat([best] * len(TRAJECTORY_MONTHS), ignore_index=True)
    traj["months"] = list(TRAJECTORY_MONTHS.values())
    traj_pred = predict_post(bundle, traj)["post_M"].values

    minus = p["pre_cyl"] <= 0
    ps, pcyl, pax = optics.from_vectors(bp["post_M"], bp["post_J0"], bp["post_J45"], minus_cyl=minus)
    plan_s, plan_c = float(best["plan_sph"].iloc[0]), float(best["plan_cyl"].iloc[0])
    ie_diff = None
    plan_ax = (int(round(float(best["plan_axis"].iloc[0]))) % 180 or 180) if plan_c else 0
    ora = 2 * float(np.hypot(pj0 - best_c[0], pj45 - best_c[1])) if cornea["usable"] else np.nan
    depth = float(optics.ablation_depth_um(plan_s, plan_c, p["oz"]))
    rsb = p["pachy"] - p["flap"] - depth
    pta = (p["flap"] + depth) / p["pachy"]

    warn = _warnings(p, bundle, pc)
    if cornea["note"]:
        warn.append(cornea["note"])
    if not np.isnan(ora) and ora > pc.get("max_ora", 1.25):
        warn.append(f"рефракционный и роговичный астигматизм расходятся на {ora:.2f} D (ORA) — "
                    "перепроверьте рефракцию и кератометрию, возможен остаточный астигматизм")
    if plan_c and p["pre_cyl"] and abs(optics.axis_diff(p["pre_axis"], plan_ax)) >= pc.get("axis_shift_warn", 15):
        warn.append(f"ось ввода {plan_ax}° отличается от оси субъективной рефракции {p['pre_axis']:.0f}° "
                    "(смещена к оси роговицы) — проверьте")
    inc = pc.get("cyl_increase", pc["cyl_search"])
    if p["pre_cyl"] and abs(plan_c) - abs(p["pre_cyl"]) >= inc - 1e-9:
        warn.append(f"цилиндр ввода увеличен на максимально допустимые {inc:g} D — "
                    "модель ожидает недокоррекцию цилиндра, решение за хирургом")
    if plan_c and not p["pre_cyl"]:
        warn.append("субъективного цилиндра нет, цилиндр ввода предложен по роговичному астигматизму")
    if ie_only:
        warn.append("субъективная рефракция не указана — расчёт выполнен от рефракции InnovEyes, "
                    "а модель обучена на субъективной")
    elif not np.isnan(p["ie_sph"]):
        im, ij0, ij45 = optics.to_vectors(p["ie_sph"], 0.0 if np.isnan(p["ie_cyl"]) else p["ie_cyl"],
                                          0.0 if np.isnan(p["ie_axis"]) else p["ie_axis"])
        dm, dc = float(im - (p["pre_sph"] + p["pre_cyl"] / 2)), 2 * float(np.hypot(ij0 - pj0, ij45 - pj45))
        ie_diff = (dm, dc)
        lim = pc.get("ie_max_diff", 0.5)
        if abs(dm) >= lim or dc >= lim:
            warn.append(f"InnovEyes и субъективная рефракция расходятся: сф. экв. {dm:+.2f} D, "
                        f"цилиндр (вектор) {dc:.2f} D — уточните рефракцию")
    if not np.isnan(rsb) and rsb < pc["min_rsb_um"]:
        warn.append(f"остаточное стромальное ложе ≈ {rsb:.0f} мкм < {pc['min_rsb_um']}")
    if not np.isnan(pta) and pta > pc["max_pta"]:
        warn.append(f"PTA ≈ {pta:.0%} > {pc['max_pta']:.0%}")
    if abs(plan_s - p["pre_sph"]) >= pc["sph_search"] - 1e-9:
        warn.append("поправка сферы упёрлась в границу поиска — проверьте данные")
    if np.isnan(p["pre_cdva"]):
        warn.append("не указана острота с коррекцией — прогноз остроты менее точен")
    elif p["pre_cdva"] < 0.8:
        warn.append(f"острота с коррекцией до операции {p['pre_cdva']:g} — ожидаемая острота без коррекции ограничена")

    def ucva(row):
        if "post_ucva_logmar" not in row:
            return None
        v = min(float(to_decimal(row["post_ucva_logmar"])), bundle.get("ucva_cap", 1.0))
        return round(v * 20) / 20

    result = {
        "id": p.get("id"), "глаз": p.get("eye"),
        "субъективно": ("не указана — расчёт от InnovEyes" if ie_only else
                        f"{p['pre_sph']:+.2f} {p['pre_cyl']:+.2f} x {p['pre_axis']:.0f}"),
        "ВВОД sph": plan_s, "ВВОД cyl": plan_c, "ВВОД axis": plan_ax,
        "поправка sph": plan_s - p["pre_sph"], "поправка cyl": plan_c - p["pre_cyl"],
        "поправка оси": round(optics.axis_diff(p["pre_axis"], plan_ax)) if plan_c and p["pre_cyl"] else None,
        "роговичный астигматизм, D": None if np.isnan(cornea["cyl"]) else round(float(cornea["cyl"]), 2),
        "плоская ось роговицы": None if not _valid(cornea["flat_axis"]) else int(round(cornea["flat_axis"])) % 180 or 180,
        "ORA, D": None if np.isnan(ora) else round(ora, 2),
        "InnovEyes": (None if np.isnan(p["ie_sph"]) else
                      f"{p['ie_sph']:+.2f} {np.nan_to_num(p['ie_cyl']):+.2f} x {np.nan_to_num(p['ie_axis']):.0f}"),
        "InnovEyes − субъективная": None if ie_diff is None else f"сф. экв. {ie_diff[0]:+.2f}, цилиндр {ie_diff[1]:.2f}",
        "цель M": target, "срок, мес": horizon,
        "отклонение прогноза от цели": round(float(bp["post_M"]) - target, 2),
        "критерий подбора": criterion,
        "прогноз M": round(float(bp["post_M"]), 2),
        "прогноз рефракции": f"{float(ps):+.2f} {float(pcyl):+.2f} x {float(pax):.0f}",
        "прогноз остаточного цилиндра": round(abs(float(pcyl)), 2),
        f"{bundle['uncertainty']['level']:.0%} глаз в пределах ± D": round(bundle["uncertainty"]["M"], 2),
        "острота с коррекцией до операции": None if np.isnan(p["pre_cdva"]) else p["pre_cdva"],
        "прогноз UCVA": ucva(bp),
        "прогноз M при вводе субъективной": round(float(mp["post_M"]), 2),
        "прогноз остаточного цилиндра при вводе субъективной":
            round(2 * float(np.hypot(mp["post_J0"], mp["post_J45"])), 2),
        "прогноз UCVA при вводе субъективной": ucva(mp),
    }
    for (name, _), v in zip(TRAJECTORY_MONTHS.items(), traj_pred):
        result[f"M {name}"] = round(float(v), 2)
    result.update({
        "абляция ≈ мкм (Мюнерлин)": round(depth), "RSB ≈ мкм": None if np.isnan(rsb) else round(rsb),
        "PTA ≈": None if np.isnan(pta) else round(pta, 3),
        "предупреждения": "; ".join(warn),
    })
    return result


def plan_table(bundle, patients, cfg, horizon=None):
    return pd.DataFrame([plan_eye(bundle, row, cfg, horizon) for row in patients.to_dict("records")])


def nomogram_tables(bundle, cfg, protocol):
    """Сводные таблицы номограммы для 'типичного' глаза (медианы биометрии обучающей выборки).

    Роговичный астигматизм типичного глаза принят равным рефракционному и на той же оси (180°):
    в таблице сферы роговица сферическая, в таблице цилиндра K2 − K1 = |цилиндр|.
    """
    tc = cfg["table"]
    base = bundle["typical"]
    k_mean = (base["k1"] + base["k2"]) / 2

    def eye(cyl, **kw):
        return dict(base, protocol=protocol, eye="OD", pre_axis=180, pre_cyl=cyl,
                    k1=k_mean - abs(cyl) / 2, k2=k_mean + abs(cyl) / 2,
                    k1_axis=180 if cyl <= 0 else 90, k2_axis=90 if cyl <= 0 else 180, **kw)
    rows = []
    for s in tc["sph_values"]:
        row = {"субъективная сфера": s}
        for age in tc["ages"]:
            r = plan_eye(bundle, eye(0.0, age=age, pre_sph=s), cfg)
            row[f"{age} лет"] = r["ВВОД sph"]
        row["предупреждения (30 лет)"] = plan_eye(bundle, eye(0.0, age=30, pre_sph=s), cfg)["предупреждения"]
        rows.append(row)
    sph_table = pd.DataFrame(rows)

    rows = []
    for c in tc["cyl_values"]:
        row = {"субъективный цилиндр": c}
        for s in tc["cyl_table_sph"]:
            r = plan_eye(bundle, eye(c, age=30, pre_sph=s), cfg)
            row[f"сфера {s:+.2f}: ввод"] = f"{r['ВВОД sph']:+.2f} {r['ВВОД cyl']:+.2f}"
        rows.append(row)
    cyl_table = pd.DataFrame(rows)
    return sph_table, cyl_table
