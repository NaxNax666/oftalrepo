"""База пациентов в Excel: запись новых пациентов из расчёта, данных операции и контрольных осмотров."""
import shutil
import threading
from datetime import date, datetime, timedelta

import numpy as np
import openpyxl
import pandas as pd

from .data import (PRE_NUMERIC, TEXT_FIELDS, VISIT_FIELDS, _extract, get_sheet, norm, norm_header,
                   read_sheet, sheet_keys, visit_names)

STATUS_PLANNED = "запланирован"
STATUS_OPERATED = "оперирован"
PRE_FIELDS = ["age", "pre_sph", "pre_cyl", "pre_axis", "pre_cdva", "k1", "k1_axis", "k2", "k2_axis",
              "al", "acd", "pachy", "pupil"]
IE_FIELDS = ["ie_sph", "ie_cyl", "ie_axis"]   # рефракция InnovEyes (протокол WLP)
OP_FIELDS = ["plan_sph", "plan_cyl", "plan_axis", "protocol", "oz", "flap", "surgery_date"]
TARGET_FIELDS = ["target_M", "target_months"]
DATE_FMT = "%d.%m.%Y"

_lock = threading.Lock()


def parse_date(v):
    if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == "":
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    s = str(v).strip()
    for fmt in (DATE_FMT, "%Y-%m-%d", "%d.%m.%y", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    raise ValueError(f"дата '{v}' не распознана (нужно ДД.ММ.ГГГГ)")


def _clean(v):
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, (datetime, date)):
        return v.strftime(DATE_FMT)
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


class Registry:
    def __init__(self, cfg):
        self.cfg = cfg
        self.path = cfg.root / cfg["data"].get("base_file", "номограмма.xlsx")
        self.header_rows = cfg["data"].get("header_rows", 2)
        self.sheet = cfg["data"].get("sheet", 0)

    # ---------- соответствие полей и колонок ----------
    def field_header(self, field):
        """Заголовок колонки ("группа|подзаголовок") для внутреннего поля, включая поля визитов v{i}_{f}."""
        if field in self.cfg["columns"]:
            return self.cfg["columns"][field]
        if field.startswith("v") and "_" in field:
            i, f = field[1:].split("_", 1)
            return f"{visit_names(self.cfg)[int(i)]}|{self.cfg['visit_columns'][f]}"
        raise KeyError(field)

    def _columns(self, ws, fields, create=None):
        """Номера колонок для полей; недостающие колонки добавляются в конец таблицы.

        Если задано create — новые колонки создаются только для этих полей (остальные пропускаются).
        """
        keys = sheet_keys(ws, self.header_rows)
        index = {k: i + 1 for i, k in enumerate(keys)}
        cols = {}
        for f in fields:
            header = self.field_header(f)
            key = norm_header(header)
            if key not in index:
                if create is not None and f not in create:
                    continue
                col = ws.max_column + 1
                parts = [p.strip() for p in header.split("|")]
                for r, part in enumerate(parts[: self.header_rows]):
                    ws.cell(r + 1, col, part)
                index[key] = col
            cols[f] = index[key]
        return cols

    def _find_rows(self, ws, id_col, eye_col):
        rows, last = {}, self.header_rows
        aliases = {k.lower(): v for k, v in self.cfg.get("aliases", {}).get("eye", {}).items()}
        for r in range(self.header_rows + 1, ws.max_row + 1):
            pid = ws.cell(r, id_col).value
            if pid is None or str(pid).strip() == "":
                continue
            last = r
            eye = ws.cell(r, eye_col).value
            eye = aliases.get(norm(eye), str(eye).strip().upper())
            rows[(str(pid).strip(), eye)] = r
        return rows, last

    def _backup(self):
        folder = self.cfg.root / "резервные_копии"
        folder.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        shutil.copy2(self.path, folder / f"{self.path.stem}_{stamp}{self.path.suffix}")
        copies = sorted(folder.glob(f"{self.path.stem}_*{self.path.suffix}"))
        for old in copies[: max(0, len(copies) - self.cfg["data"].get("backups_keep", 50))]:
            old.unlink()

    def write(self, updates, only_if=None):
        """updates: список (id, глаз, {поле: значение}). Строки создаются, если их нет.

        only_if(existing_status) -> bool: если задано и вернуло False — строка не меняется.
        Возвращает список (id, глаз, "создан"/"обновлён"/"пропущен: причина").
        """
        with _lock:
            wb = openpyxl.load_workbook(self.path)
            ws = get_sheet(wb, self.sheet)
            order = {f: i for i, f in enumerate(self.cfg["columns"])}   # новые колонки — в порядке config.toml
            fields = sorted({f for _, _, vals in updates for f in vals} | {"id", "eye", "status"},
                            key=lambda f: (order.get(f, len(order)), f))
            # новые колонки (например, InnovEyes) добавляются в таблицу, только когда в них есть что записать
            filled = {f for _, _, vals in updates for f, v in vals.items() if v is not None} | {"id", "eye", "status"}
            cols = self._columns(ws, fields, create=filled)
            rows, last = self._find_rows(ws, cols["id"], cols["eye"])
            report = []
            for pid, eye, vals in updates:
                key = (str(pid).strip(), eye)
                r = rows.get(key)
                if r is not None and only_if is not None:
                    status = norm(ws.cell(r, cols["status"]).value)
                    if not only_if(status):
                        report.append((pid, eye, "пропущен: пациент уже прооперирован, данные не перезаписаны"))
                        continue
                if r is None:
                    last += 1
                    r = rows[key] = last
                    ws.cell(r, cols["id"], str(pid).strip())
                    ws.cell(r, cols["eye"], eye)
                    report.append((pid, eye, "создан"))
                else:
                    report.append((pid, eye, "обновлён"))
                for f, v in vals.items():
                    if f in cols:
                        ws.cell(r, cols[f], v)
            self._backup()
            wb.save(self.path)
        return report

    # ---------- чтение ----------
    def frame(self):
        raw = read_sheet(self.path, self.sheet, self.header_rows)
        df = _extract(raw, self.cfg, None)
        df["id"] = df["id"].astype(str).str.strip()
        aliases = {k.lower(): v for k, v in self.cfg.get("aliases", {}).get("eye", {}).items()}
        df["eye"] = df["eye"].map(lambda v: aliases.get(norm(v), str(v).strip().upper()))
        return df

    def _visit_state(self, row, today):
        names, months = visit_names(self.cfg), list(self.cfg["visits"].values())
        surgery = None
        try:
            surgery = parse_date(row.get("surgery_date"))
        except ValueError:
            pass
        done, due = [], []
        for i, (name, mo) in enumerate(zip(names, months)):
            filled = pd.notna(row.get(f"v{i}_sph")) or pd.notna(row.get(f"v{i}_ucva"))
            if filled:
                done.append(name)
            elif surgery is not None:
                when = surgery + timedelta(days=round(mo * 30.4))
                if when <= today:
                    due.append((name, when))
        return surgery, done, due

    def patients(self):
        df = self.frame()
        today = date.today()
        out = []
        for pid, g in df.groupby("id", sort=False):
            eyes, statuses, done_all, due_all, surgery = [], set(), {}, [], None
            for _, row in g.iterrows():
                st = norm(row.get("status")) or STATUS_OPERATED
                statuses.add(st)
                s, done, due = self._visit_state(row, today)
                surgery = surgery or s
                eyes.append(row["eye"])
                for name in done:
                    done_all[name] = done_all.get(name, 0) + 1
                due_all += [f"{row['eye']} {n}" for n, _ in due]
            status = STATUS_PLANNED if STATUS_PLANNED in statuses else STATUS_OPERATED
            out.append({
                "id": pid, "age": _clean(g["age"].iloc[0]), "eyes": " ".join(eyes), "status": status,
                "surgery_date": surgery.strftime(DATE_FMT) if surgery else None,
                "calc_date": _clean(g["calc_date"].dropna().iloc[0]) if g["calc_date"].notna().any() else None,
                "visits": [n for n in visit_names(self.cfg) if n in done_all],
                "due": due_all, "_row": int(g["_row"].min()),
            })
        out.sort(key=lambda p: -p["_row"])
        return out

    def patient(self, pid):
        df = self.frame()
        g = df[df["id"] == str(pid).strip()]
        if g.empty:
            raise ValueError(f"Пациент '{pid}' не найден в базе")
        fields = ["age", *PRE_FIELDS[1:], *IE_FIELDS, *OP_FIELDS, *TARGET_FIELDS,
                  "comment", "status", "calc_date", "recommendation"]
        visits = [f"v{i}_{f}" for i in range(len(visit_names(self.cfg))) for f in VISIT_FIELDS]
        eyes = {}
        for _, row in g.iterrows():
            eyes[row["eye"]] = {f: _clean(row.get(f)) for f in fields + visits}
            eyes[row["eye"]]["status"] = norm(row.get("status")) or STATUS_OPERATED
            eyes[row["eye"]]["_row"] = int(row["_row"])
        return {"id": str(pid).strip(), "eyes": eyes, "file": self.path.name}

    def lookup(self, pid):
        """Есть ли пациент в базе (сравнение без учёта регистра и лишних пробелов)."""
        key = norm(pid)
        if not key:
            return {"exists": False}
        df = self.frame()
        g = df[df["id"].map(norm) == key]
        if g.empty:
            return {"exists": False}
        eyes = []
        for _, row in g.iterrows():
            surgery = None
            try:
                surgery = parse_date(row.get("surgery_date"))
            except ValueError:
                pass
            eyes.append({"eye": row["eye"], "status": norm(row.get("status")) or STATUS_OPERATED,
                         "calc_date": _clean(row.get("calc_date")), "protocol": _clean(row.get("protocol")),
                         "surgery_date": surgery.strftime(DATE_FMT) if surgery else None,
                         "row": int(row["_row"])})
        return {"exists": True, "id": g["id"].iloc[0], "age": _clean(g["age"].iloc[0]), "eyes": eyes,
                "file": self.path.name}


def validate(cfg, values):
    """Преобразует введённые значения к числам и проверяет диапазоны. Возвращает (чистые значения, ошибки)."""
    ranges = cfg["validation"]["ranges"]
    clean, errors = {}, []
    for f, v in values.items():
        if f in TEXT_FIELDS:
            s = "" if v is None else str(v).strip()
            if f == "protocol" and s:
                aliases = {k.lower(): v for k, v in cfg.get("aliases", {}).get("protocol", {}).items()}
                s = aliases.get(norm(s), s)
            if f == "surgery_date" and s:
                try:
                    s = parse_date(s).strftime(DATE_FMT)
                except ValueError as e:
                    errors.append(str(e))
            clean[f] = s or None
            continue
        s = "" if v is None else str(v).strip().replace(",", ".")
        if s == "":
            clean[f] = None
            continue
        try:
            x = float(s)
        except ValueError:
            errors.append(f"{f}: '{v}' — не число")
            continue
        key = f if f in ranges else (f"visit_{f.split('_', 1)[1]}" if f.startswith("v") and "_" in f else None)
        if key in ranges and not ranges[key][0] <= x <= ranges[key][1]:
            errors.append(f"{f}: {x:g} вне допустимого диапазона [{ranges[key][0]}; {ranges[key][1]}]")
            continue
        clean[f] = int(x) if x.is_integer() and (f.endswith("axis") or f in ("age", "pachy", "flap",
                                                                               "target_months")) else x
    return clean, errors


def planned_record(patient, eye_input, result, version):
    """Поля новой строки базы по данным расчёта (ввод в лазер = рекомендация программы)."""
    vals = {f: eye_input.get(f) for f in PRE_FIELDS}
    vals["age"] = patient.get("age")
    if any(str(eye_input.get(f) or "").strip() for f in IE_FIELDS):   # колонки InnovEyes — только если заполнены
        vals.update({f: eye_input.get(f) for f in IE_FIELDS})
    vals.update({"target_M": result["цель M"], "target_months": result["срок, мес"]})
    vals.update({
        "plan_sph": result["ВВОД sph"], "plan_cyl": result["ВВОД cyl"], "plan_axis": result["ВВОД axis"],
        "protocol": patient.get("protocol"), "oz": patient.get("oz"), "flap": eye_input.get("flap"),
        "status": STATUS_PLANNED, "calc_date": datetime.now().strftime(DATE_FMT),
        "recommendation": (f"{result['ВВОД sph']:+.2f} {result['ВВОД cyl']:+.2f} x {result['ВВОД axis']}; "
                           f"цель {result['цель M']:+.2f}, прогноз M {result['прогноз M']:+.2f} "
                           f"через {result['срок, мес']:g} мес; "
                           f"модель {version}"),
    })
    return vals
