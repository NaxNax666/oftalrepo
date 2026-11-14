"""Локальный веб-интерфейс: http://127.0.0.1:8765 (без внешних зависимостей, только стандартная библиотека)."""
import json
import math
import re
import tempfile
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

import numpy as np
import pandas as pd

from . import cli
from .data import data_files, load_dataset
from .planner import nomogram_tables, plan_eye, plan_table
from .data import norm
from .registry import IE_FIELDS, STATUS_OPERATED, STATUS_PLANNED, Registry, planned_record, validate
from .training import load_bundle, train, write_excel

WEB_DIR = Path(__file__).resolve().parent / "web"
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def to_json(obj):
    if isinstance(obj, dict):
        return {str(k): to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_json(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return to_json(obj.to_dict("records"))
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if obj is pd.NA or obj is pd.NaT:
        return None
    return obj


def number(v):
    if v is None:
        return None
    s = str(v).strip().replace(",", ".")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        raise ValueError(f"'{v}' — не число")


class App:
    def __init__(self, cfg):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.bundle = None
        self.registry = Registry(cfg)
        self.reload()

    def reload(self):
        try:
            self.bundle = load_bundle(self.cfg)
        except FileNotFoundError:
            self.bundle = None

    def need_model(self):
        if self.bundle is None:
            raise ValueError("Модель ещё не обучена. Откройте вкладку «Данные и обучение» и нажмите «Переобучить».")
        return self.bundle

    def status(self):
        b = self.bundle
        pc = self.cfg["planning"]
        files = [Path(p).name for p in data_files(self.cfg)] if self._has_files() else []
        extra = sorted(p.name for p in (self.cfg.root / "данные").glob("*.xlsx")) \
            if (self.cfg.root / "данные").exists() else []
        out = {"trained": b is not None, "files": files, "uploaded": extra,
               "planning": {"target_M": pc["target_M"], "horizon_months": pc["horizon_months"],
                            "default_flap": pc["default_flap"]}}
        trained = list(b["protocols"]) if b else []
        out["all_protocols"] = trained + [p for p in pc.get("protocols", []) if p not in trained]
        if b:
            hist_path = self.cfg.models_dir / "история_обучения.csv"
            out.update({
                "version": b["version"], "trained_at": b.get("trained_at", b["version"]),
                "n_eyes": b["n_eyes"], "n_rows": b["n_rows"], "protocols": b["protocols"],
                "protocol_counts": b.get("protocol_counts", {}),
                "typical": b["typical"], "uncertainty": b["uncertainty"],
                "summary": [{"k": k, "v": v} for k, v in b.get("summary", [])],
                "history": pd.read_csv(hist_path, encoding="utf-8-sig") if hist_path.exists() else [],
            })
        return out

    def _has_files(self):
        try:
            data_files(self.cfg)
            return True
        except FileNotFoundError:
            return False

    def plan(self, req):
        b = self.need_model()
        aliases = {k.lower(): v for k, v in self.cfg.get("aliases", {}).get("protocol", {}).items()}
        req["protocol"] = aliases.get(norm(req.get("protocol")), req.get("protocol"))
        wlp = req["protocol"] == "WLP"
        common = {"id": req.get("id") or "", "age": number(req.get("age")),
                  "protocol": req.get("protocol"), "oz": number(req.get("oz")),
                  "target_M": number(req.get("target_M"))}
        horizon = number(req.get("horizon"))
        results, inputs = [], {}
        for eye in ["OD", "OS"]:
            e = dict(req.get("eyes", {}).get(eye) or {})
            if not wlp:   # рефракция InnovEyes относится только к протоколу WLP
                for k in IE_FIELDS:
                    e.pop(k, None)
            if number(e.get("pre_sph")) is None and number(e.get("ie_sph")) is None:
                continue
            p = dict(common, eye=eye)
            for k in ["pre_sph", "pre_cyl", "pre_axis", "pre_cdva", "k1", "k1_axis", "k2", "k2_axis",
                      "al", "acd", "pachy", "pupil", "flap", *IE_FIELDS]:
                p[k] = number(e.get(k))
            inputs[eye] = e
            if p["pre_cyl"] and p["pre_axis"] is None:
                raise ValueError(f"{eye}: укажите ось цилиндра")
            if p["ie_cyl"] and p["ie_axis"] is None:
                raise ValueError(f"{eye}: укажите ось цилиндра InnovEyes")
            if p["ie_sph"] is None and (p["ie_cyl"] is not None or p["ie_axis"] is not None):
                raise ValueError(f"{eye}: InnovEyes — укажите сферу")
            if p["target_M"] is None:
                p.pop("target_M")
            if p["age"] is None:
                raise ValueError("Укажите возраст")
            results.append(plan_eye(b, p, self.cfg, horizon))
        if not results:
            raise ValueError("Заполните рефракцию хотя бы для одного глаза"
                             + (" (субъективную или InnovEyes)" if wlp else ""))
        return {"results": results, "version": b["version"],
                "saved": self._save_planned(req, inputs, results, b["version"])}

    def _save_planned(self, req, inputs, results, version):
        pid = str(req.get("id") or "").strip()
        if not pid:
            return {"ok": False, "message": "В базу не сохранено: не указан № карты / ФИО"}
        updates, errors = [], []
        for r in results:
            vals = planned_record(req, inputs[r["глаз"]], r, version)
            clean, err = validate(self.cfg, vals)
            errors += [f"{r['глаз']}: {e}" for e in err]
            updates.append((pid, r["глаз"], clean))
        if errors:
            return {"ok": False, "message": "В базу не сохранено: " + "; ".join(errors)}
        try:
            report = self.registry.write(updates, only_if=lambda status: status == STATUS_PLANNED)
        except PermissionError:
            return {"ok": False, "message": f"В базу не сохранено: файл {self.registry.path.name} "
                                            "открыт в Excel — закройте его и нажмите «Рассчитать» ещё раз"}
        return {"ok": True, "message": "; ".join(f"{eye}: {msg}" for _, eye, msg in report),
                "id": pid, "file": self.registry.path.name}

    def patients(self):
        return {"patients": self.registry.patients(), "visits": list(self.cfg["visits"].keys()),
                "file": self.registry.path.name}

    def patient(self, pid):
        return self.registry.patient(pid)

    def patient_exists(self, pid):
        return self.registry.lookup(pid)

    def save_patient(self, req):
        pid = str(req.get("id") or "").strip()
        if not pid:
            raise ValueError("Не указан пациент")
        n_visits = len(self.cfg["visits"])
        updates, errors = [], []
        for eye, e in (req.get("eyes") or {}).items():
            vals = {k: v for k, v in e.items() if k not in ("status", "operated", "_row")}
            clean, err = validate(self.cfg, vals)
            errs = [f"{eye}: {x}" for x in err]
            has_visit = any(clean.get(f"v{i}_{f}") is not None
                            for i in range(n_visits) for f in ("sph", "ucva"))
            if e.get("operated"):
                if clean.get("plan_sph") is None:
                    errs.append(f"{eye}: укажите фактический ввод в лазер (сфера)")
                if clean.get("flap") is None:
                    errs.append(f"{eye}: укажите толщину лоскута, измеренную в операционной")
                clean["status"] = STATUS_OPERATED
            else:
                if has_visit:
                    errs.append(f"{eye}: есть контрольные осмотры — отметьте «Операция проведена»")
                clean["status"] = STATUS_PLANNED
            errors += errs
            updates.append((pid, eye, clean))
        if errors:
            raise ValueError("Не сохранено:\n" + "\n".join(errors))
        self.registry.write(updates)
        return {"ok": True, "patient": self.registry.patient(pid)}

    def plan_file(self, body, horizon=None):
        b = self.need_model()
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.xlsx"
            src.write_bytes(body)
            patients = cli.read_patients(src)
            if patients.empty:
                raise ValueError("В файле не найдено пациентов (колонка sph пуста или шаблон другой)")
            aliases = {k.lower(): v for k, v in self.cfg.get("aliases", {}).get("protocol", {}).items()}
            patients["protocol"] = patients["protocol"].map(lambda v: aliases.get(str(v).strip().lower(), v))
            result = plan_table(b, patients, self.cfg, horizon)
            out = Path(tmp) / "out.xlsx"
            write_excel(out, {"План": result})
            return out.read_bytes()

    def check(self):
        wide, issues = load_dataset(self.cfg)
        return {"eyes": len(wide), "excluded": int((wide["excluded"] != "").sum()),
                "issues": issues.frame()}

    def train(self):
        if not self.lock.acquire(blocking=False):
            raise ValueError("Обучение уже идёт, подождите")
        try:
            bundle, summary, issues = train(self.cfg, log=lambda *a: None)
            self.bundle = bundle
            return {"summary": [{"k": k, "v": v} for k, v in summary], "issues": issues.frame()}
        finally:
            self.lock.release()

    def upload_data(self, name, body):
        name = re.sub(r'[\\/:*?"<>|]', "_", Path(name or "данные.xlsx").name)
        if not name.lower().endswith(".xlsx"):
            raise ValueError("Нужен файл Excel .xlsx")
        folder = self.cfg.root / "данные"
        folder.mkdir(exist_ok=True)
        path = folder / name
        path.write_bytes(body)
        try:
            result = self.check()
        except Exception as e:
            path.unlink()
            raise ValueError(f"Файл не подходит по формату. {e}")
        result["saved"] = name
        return result

    def delete_data(self, name):
        path = self.cfg.root / "данные" / Path(name).name
        if path.exists():
            path.unlink()
        return {"deleted": name}

    def tables(self, protocol):
        b = self.need_model()
        sph, cyl = nomogram_tables(b, self.cfg, protocol or b["protocols"][0])
        return {"sph": sph, "cyl": cyl}

    def template(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "t.xlsx"
            write_excel(out, {"Пациенты": pd.DataFrame([cli.EXAMPLE], columns=list(cli.PLAN_COLUMNS.values()))})
            return out.read_bytes()

    def result_file(self, name):
        path = self.cfg.results_dir / Path(name).name
        if name == "история":
            path = self.cfg.models_dir / "история_обучения.csv"
        if not path.exists():
            raise ValueError("Файл не найден — сначала обучите модель")
        return path


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(self, code, body, ctype, filename=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if filename:
                self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{quote(filename)}")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(to_json(obj), ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(n) if n else b""

        def _run(self, fn):
            try:
                fn()
            except ValueError as e:
                self._json({"error": str(e)}, 400)
            except PermissionError:
                self._json({"error": "Файл занят другой программой (закройте его в Excel) и повторите"}, 400)
            except Exception as e:
                traceback.print_exc()
                self._json({"error": f"Внутренняя ошибка: {e}"}, 500)

        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}

            def go():
                if url.path in ("/", "/index.html"):
                    self._send(200, (WEB_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
                elif url.path == "/favicon.ico":
                    self._send(204, b"", "image/x-icon")
                elif url.path == "/api/status":
                    self._json(app.status())
                elif url.path == "/api/patients":
                    self._json(app.patients())
                elif url.path == "/api/patient":
                    self._json(app.patient(q.get("id", "")))
                elif url.path == "/api/patient-exists":
                    self._json(app.patient_exists(q.get("id", "")))
                elif url.path == "/api/tables":
                    self._json(app.tables(q.get("protocol")))
                elif url.path == "/api/template":
                    self._send(200, app.template(), XLSX, "шаблон_планирования.xlsx")
                elif url.path == "/api/download":
                    path = app.result_file(q.get("name", ""))
                    ctype = "text/csv; charset=utf-8" if path.suffix == ".csv" else XLSX
                    self._send(200, path.read_bytes(), ctype, path.name)
                else:
                    self._send(404, b"not found", "text/plain")
            self._run(go)

        def do_POST(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}

            def go():
                if url.path == "/api/plan":
                    self._json(app.plan(json.loads(self._body() or b"{}")))
                elif url.path == "/api/plan-file":
                    horizon = number(q.get("horizon"))
                    self._send(200, app.plan_file(self._body(), horizon), XLSX, "план_операций.xlsx")
                elif url.path == "/api/patient/save":
                    self._json(app.save_patient(json.loads(self._body() or b"{}")))
                elif url.path == "/api/check":
                    self._json(app.check())
                elif url.path == "/api/train":
                    self._json(app.train())
                elif url.path == "/api/upload-data":
                    self._json(app.upload_data(q.get("name"), self._body()))
                elif url.path == "/api/delete-data":
                    self._json(app.delete_data(q.get("name", "")))
                else:
                    self._send(404, b"not found", "text/plain")
            self._run(go)

    return Handler


def serve(cfg, host="127.0.0.1", port=8765, open_browser=True):
    url = f"http://{host}:{port}/"
    try:
        httpd = ThreadingHTTPServer((host, port), make_handler(App(cfg)))
    except OSError:
        print(f"Программа уже запущена (или порт {port} занят). Открываю {url}")
        if open_browser:
            webbrowser.open(url)
        return
    print("=" * 60)
    print(" Номограмма LASIK (Alcon EX500 + Moria M2)")
    print(f" Откройте в браузере: {url}")
    print(" Чтобы остановить программу — закройте это окно (или Ctrl+C)")
    print("=" * 60, flush=True)
    if open_browser:
        threading.Timer(1.0, webbrowser.open, args=[url]).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
