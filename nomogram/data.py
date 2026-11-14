"""Чтение Excel-таблиц, проверка качества данных и подготовка обучающей выборки."""
import glob
import re
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

from . import optics

PRE_NUMERIC = [
    "age", "pre_sph", "pre_cyl", "pre_axis", "pre_cdva", "k1", "k1_axis", "k2", "k2_axis",
    "al", "acd", "pachy", "pupil", "plan_sph", "plan_cyl", "plan_axis", "oz", "flap",
    "ie_sph", "ie_cyl", "ie_axis", "target_M", "target_months",
]
REQUIRED = ["id", "eye", "pre_sph", "pre_cyl", "plan_sph", "plan_cyl"]
TEXT_FIELDS = ["id", "eye", "protocol", "comment", "status", "surgery_date", "calc_date", "recommendation"]
VISIT_FIELDS = ["sph", "cyl", "axis", "ucva"]


def norm(value):
    if value is None or value is pd.NA or (isinstance(value, float) and value != value):
        return ""   # пустая ячейка (в т.ч. NaN из pandas) — пустая строка, а не "nan"
    s = str(value).strip().lower().replace("ё", "е")
    return re.sub(r"\s+", " ", s)


def norm_header(header):
    return "|".join(norm(p) for p in str(header).split("|") if norm(p))


def _unique(keys):
    seen, out = {}, []
    for k in keys:
        seen[k] = seen.get(k, 0) + 1
        out.append(k if seen[k] == 1 else f"{k}#{seen[k]}")
    return out


def get_sheet(wb, sheet):
    return wb.worksheets[sheet] if isinstance(sheet, int) else wb[sheet]


def sheet_keys(ws, header_rows=2):
    """Ключи колонок листа с многоуровневой шапкой: "группа|подзаголовок"."""
    ncol = ws.max_column
    grid = [[ws.cell(r, c).value for c in range(1, ncol + 1)] for r in range(1, header_rows + 1)]
    for rng in ws.merged_cells.ranges:
        if rng.min_row > header_rows:
            continue
        value = ws.cell(rng.min_row, rng.min_col).value
        for r in range(rng.min_row, min(rng.max_row, header_rows) + 1):
            for c in range(rng.min_col, rng.max_col + 1):
                grid[r - 1][c - 1] = value
    keys = []
    for c in range(ncol):
        parts = []
        for r in range(header_rows):
            p = norm(grid[r][c])
            if p and p not in parts:
                parts.append(p)
        keys.append("|".join(parts) or f"col{c + 1}")
    return _unique(keys)


def read_sheet(path, sheet=0, header_rows=2):
    """Читает лист с многоуровневой шапкой; колонки называются "группа|подзаголовок"."""
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = get_sheet(wb, sheet)
    keys = sheet_keys(ws, header_rows)
    rows = list(ws.iter_rows(min_row=header_rows + 1, max_col=len(keys), values_only=True))
    df = pd.DataFrame(rows, columns=keys)
    df["_file"] = Path(path).name
    df["_row"] = np.arange(len(df)) + header_rows + 1
    data_cols = [c for c in df.columns if not c.startswith("_")]
    return df[df[data_cols].notna().any(axis=1)].reset_index(drop=True)


class Issues(list):
    def add(self, row, field, value, problem, action):
        self.append({
            "файл": row.get("_file"), "строка": row.get("_row"), "id": row.get("id"),
            "глаз": row.get("eye"), "поле": field, "значение": value,
            "проблема": problem, "действие": action,
        })

    def frame(self):
        cols = ["файл", "строка", "id", "глаз", "поле", "значение", "проблема", "действие"]
        return pd.DataFrame(self, columns=cols)


def visit_names(cfg):
    return list(cfg["visits"].keys())


def _extract(raw, cfg, issues):
    """Переименовывает колонки таблицы во внутренние имена полей."""
    out = pd.DataFrame(index=raw.index)
    out["_file"], out["_row"] = raw["_file"], raw["_row"]
    columns = dict(cfg["columns"])
    for field in PRE_NUMERIC + TEXT_FIELDS:   # поля, не описанные в старом config.toml, считаются пустыми
        columns.setdefault(field, f"__нет__{field}")
    for field, header in columns.items():
        key = norm_header(header)
        if key in raw.columns:
            out[field] = raw[key]
        else:
            if field in REQUIRED:
                raise KeyError(f"нет обязательной колонки '{header}' (шапка должна быть как в номограмма.xlsx)")
            out[field] = np.nan
    vc = cfg["visit_columns"]
    for i, vname in enumerate(visit_names(cfg)):
        for f in VISIT_FIELDS:
            key = norm_header(f"{vname}|{vc[f]}")
            out[f"v{i}_{f}"] = raw[key] if key in raw.columns else np.nan
    return out


def _to_number(df, col, issues):
    parsed = pd.to_numeric(
        df[col].map(lambda v: str(v).replace(",", ".").strip() if isinstance(v, str) else v),
        errors="coerce",
    )
    bad = df[col].notna() & parsed.isna() & (df[col].astype(str).str.strip() != "")
    for idx in df.index[bad]:
        issues.add(df.loc[idx], col, df.at[idx, col], "не число", "значение не используется")
    df[col] = parsed


def _check_range(df, col, lo, hi, issues, label=None):
    bad = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
    for idx in df.index[bad]:
        issues.add(df.loc[idx], label or col, df.at[idx, col],
                   f"вне диапазона [{lo}; {hi}]", "значение не используется")
    df.loc[bad, col] = np.nan


def clean(wide, cfg, issues):
    df = wide.copy()
    df["id"] = df["id"].astype(str).str.strip()
    for field, aliases in cfg.get("aliases", {}).items():
        if field in df:
            lower = {k.lower(): v for k, v in aliases.items()}
            df[field] = df[field].map(
                lambda v: lower.get(norm(v), str(v).strip()) if pd.notna(v) else v)

    extra_numeric = [f for f in cfg["columns"] if f not in PRE_NUMERIC + TEXT_FIELDS]
    for col in PRE_NUMERIC + extra_numeric:
        _to_number(df, col, issues)
    ranges = cfg["validation"]["ranges"]
    for col in PRE_NUMERIC + extra_numeric:
        if col in ranges:
            _check_range(df, col, *ranges[col], issues)

    names = visit_names(cfg)
    for i, vname in enumerate(names):
        for f in VISIT_FIELDS:
            col = f"v{i}_{f}"
            _to_number(df, col, issues)
            if f"visit_{f}" in ranges:
                _check_range(df, col, *ranges[f"visit_{f}"], issues, label=f"{vname}: {f}")

    for col in ["pre_axis", "plan_axis", "ie_axis"]:
        cyl = col.replace("axis", "cyl")
        df.loc[df[col].isna() & (df[cyl] == 0), col] = 0
    # InnovEyes: сфера без цилиндра — цилиндр 0
    df.loc[df["ie_sph"].notna() & df["ie_cyl"].isna(), ["ie_cyl", "ie_axis"]] = 0

    # Субъективной рефракции нет, но есть InnovEyes (WLP) — в расчётах используется рефракция InnovEyes
    ie_only = df["pre_sph"].isna() & df["ie_sph"].notna()
    for f in ["sph", "cyl", "axis"]:
        df.loc[ie_only, f"pre_{f}"] = df.loc[ie_only, f"ie_{f}"]

    df["excluded"] = ""
    for col in REQUIRED:
        miss = df[col].isna() | (df[col].astype(str).str.strip().isin(["", "nan"]))
        for idx in df.index[miss]:
            issues.add(df.loc[idx], col, None, "нет обязательного значения", "глаз исключён")
        df.loc[miss, "excluded"] = "нет обязательных данных"
    for cyl, axis in [("pre_cyl", "pre_axis"), ("plan_cyl", "plan_axis")]:
        miss = df[axis].isna() & (df[cyl].fillna(0) != 0)
        for idx in df.index[miss]:
            issues.add(df.loc[idx], axis, None, "нет оси при ненулевом цилиндре", "глаз исключён")
        df.loc[miss, "excluded"] = "нет оси цилиндра"

    if cfg["validation"].get("exclude_eyes_with_comment", True) and "comment" in df:
        has = df["comment"].notna() & (df["comment"].astype(str).str.strip() != "")
        for idx in df.index[has]:
            issues.add(df.loc[idx], "comment", df.at[idx, "comment"], "осложнение/комментарий",
                       "глаз исключён из обучения")
        df.loc[has & (df["excluded"] == ""), "excluded"] = "комментарий"

    skip_status = [norm(s) for s in cfg["validation"].get("exclude_status", [])]
    if skip_status and "status" in df:
        planned = df["status"].map(norm).isin(skip_status) & (df["excluded"] == "")
        df.loc[planned, "excluded"] = "операция не подтверждена"

    thr_m = cfg["validation"]["suspicious_postop_M"]
    thr_v = cfg["validation"]["suspicious_postop_ucva"]
    for i, vname in enumerate(names):
        s, c, a, v = (f"v{i}_{f}" for f in VISIT_FIELDS)
        m = df[s] + df[c].fillna(0) / 2
        bad = (m.abs() >= thr_m) & (df[v] >= thr_v)
        for idx in df.index[bad]:
            issues.add(df.loc[idx], f"{vname}: рефракция",
                       f"{df.at[idx, s]} / {df.at[idx, c]} x {df.at[idx, a]}, UCVA {df.at[idx, v]}",
                       f"|сф.экв.| {abs(m[idx]):.2f} D при UCVA {df.at[idx, v]} — вероятна опечатка",
                       "визит не используется")
        df.loc[bad, [s, c, a]] = np.nan

    dup = df.duplicated(["id", "eye"], keep="last")
    for idx in df.index[dup]:
        issues.add(df.loc[idx], "id+глаз", f"{df.at[idx, 'id']} {df.at[idx, 'eye']}",
                   "повтор записи", "используется последняя запись")
    return df[~dup].reset_index(drop=True)


def data_files(cfg):
    files = []
    for pattern in cfg["data"]["files"]:
        matches = sorted(glob.glob(str(cfg.root / pattern)))
        files += [m for m in matches if not Path(m).name.startswith("~$") and m not in files]
    if not files:
        raise FileNotFoundError("Не найдено ни одного файла данных (раздел [data] в config.toml)")
    return files


def load_dataset(cfg):
    """Возвращает (таблица глаз в широком формате, журнал проблем)."""
    issues = Issues()
    parts = []
    for path in data_files(cfg):
        raw = read_sheet(path, cfg["data"].get("sheet", 0), cfg["data"].get("header_rows", 2))
        try:
            parts.append(_extract(raw, cfg, issues))
        except KeyError as e:
            raise ValueError(f"Файл {Path(path).name}: {e.args[0]}") from None
    wide = pd.concat(parts, ignore_index=True)
    return clean(wide, cfg, issues), issues


def to_long(wide, cfg):
    """Один ряд = один глаз на одном визите с достоверной рефракцией."""
    rows = []
    months = list(cfg["visits"].values())
    names = visit_names(cfg)
    base = wide[wide["excluded"] == ""]
    for i, (vname, mo) in enumerate(zip(names, months)):
        s, c, a, v = (f"v{i}_{f}" for f in VISIT_FIELDS)
        ok = base[s].notna() & base[c].notna() & ((base[c] == 0) | base[a].notna())
        part = base.loc[ok].drop(columns=[x for x in base.columns if re.match(r"v\d+_", x)]).copy()
        part["visit"] = vname
        part["months"] = mo
        part["post_sph"] = base.loc[ok, s].values
        part["post_cyl"] = base.loc[ok, c].values
        part["post_axis"] = base.loc[ok, a].fillna(0).values
        part["post_ucva"] = base.loc[ok, v].values
        rows.append(part)
    long = pd.concat(rows, ignore_index=True)
    m, j0, j45 = optics.to_vectors(long["post_sph"], long["post_cyl"], long["post_axis"])
    long["post_M"], long["post_J0"], long["post_J45"] = m, j0, j45
    return long
