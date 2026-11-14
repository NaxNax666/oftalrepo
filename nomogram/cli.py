import argparse
import sys
from pathlib import Path

import pandas as pd

from .config import load_config
from .data import load_dataset, norm
from .planner import plan_table
from .training import load_bundle, train, write_excel, write_tables

PLAN_COLUMNS = {
    "id": "id", "age": "полных лет", "eye": "глаз",
    "pre_sph": "sph", "pre_cyl": "cyl", "pre_axis": "axis", "pre_cdva": "зрение с коррекцией",
    "k1": "k1", "k1_axis": "k1 axis", "k2": "k2", "k2_axis": "k2 axis",
    "al": "al", "acd": "acd", "pachy": "pahymerty", "pupil": "диамтер зрачка",
    "protocol": "протокол", "oz": "optical zone", "flap": "flap (ожидаемый)",
    "target_M": "целевая рефракция",
    # рефракция InnovEyes — только для протокола WLP (WaveLight Plus), можно не заполнять
    "ie_sph": "innoveyes sph", "ie_cyl": "innoveyes cyl", "ie_axis": "innoveyes axis",
}
EXAMPLE = ["пример", 30, "OD", -3.5, -0.75, 180, 1.0, 43.0, 180, 43.9, 90, 24.8, 3.7, 545, 5.9,
           "Custom-Q", 6.5, 110, 0, None, None, None]


def cmd_check(cfg, args):
    wide, issues = load_dataset(cfg)
    print(f"Глаз: {len(wide)}, исключено: {(wide['excluded'] != '').sum()}, замечаний: {len(issues)}")
    frame = issues.frame()
    if len(frame):
        print(frame[["строка", "id", "глаз", "поле", "значение", "проблема"]].to_string(index=False))
    out = cfg.results_dir / "проверка_данных.xlsx"
    write_excel(out, {"Качество данных": frame})
    print(f"\nОтчёт: {out}")


def cmd_train(cfg, args):
    bundle, summary, issues = train(cfg)
    for k, v in summary:
        print(f"{k}: {v}")
    print(f"\nМодель сохранена: {cfg.models_dir}\nОтчёты: {cfg.results_dir}")


def read_patients(path):
    raw = pd.read_excel(path)
    raw.columns = [norm(c) for c in raw.columns]
    df = pd.DataFrame()
    for field, label in PLAN_COLUMNS.items():
        df[field] = raw[norm(label)] if norm(label) in raw.columns else None
    df["eye"] = df["eye"].astype(str).str.upper().str.strip()
    return df[df["pre_sph"].notna() | df["ie_sph"].notna()]


def cmd_plan(cfg, args):
    bundle = load_bundle(cfg)
    patients = read_patients(args.input)
    aliases = {k.lower(): v for k, v in cfg.get("aliases", {}).get("protocol", {}).items()}
    patients["protocol"] = patients["protocol"].map(lambda v: aliases.get(norm(v), v))
    result = plan_table(bundle, patients, cfg, args.horizon)
    out = Path(args.output) if args.output else cfg.results_dir / f"план_{Path(args.input).stem}.xlsx"
    write_excel(out, {"План": result})
    cols = ["id", "глаз", "субъективно", "ВВОД sph", "ВВОД cyl", "ВВОД axis", "цель M", "прогноз M",
            "предупреждения"]
    print(result[cols].to_string(index=False))
    print(f"\nМодель {bundle['version']}. Результат: {out}")


def cmd_template(cfg, args):
    out = cfg.root / "планирование" / "шаблон_планирования.xlsx"
    write_excel(out, {"Пациенты": pd.DataFrame([EXAMPLE], columns=list(PLAN_COLUMNS.values()))})
    print(f"Шаблон: {out}")


def cmd_table(cfg, args):
    bundle = load_bundle(cfg)
    write_tables(bundle, cfg)
    print(f"Таблица: {cfg.results_dir / 'номограмма_таблица.xlsx'}")


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except AttributeError:
            pass
    ap = argparse.ArgumentParser(prog="python -m nomogram",
                                 description="Номограмма LASIK: Alcon EX500 + Moria M2")
    ap.add_argument("--config", help="путь к config.toml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="проверить данные и показать ошибки ввода")
    sub.add_parser("train", help="обучить (переобучить) модель на всех данных")
    p = sub.add_parser("plan", help="рассчитать ввод в лазер для новых пациентов")
    p.add_argument("input", help="Excel по шаблону (команда template)")
    p.add_argument("-o", "--output")
    p.add_argument("--horizon", type=float, help="срок оптимизации, мес (по умолчанию из config)")
    sub.add_parser("template", help="создать шаблон Excel для планирования")
    sub.add_parser("table", help="пересчитать сводную таблицу номограммы")
    s = sub.add_parser("serve", help="запустить веб-интерфейс в браузере")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--no-browser", action="store_true", help="не открывать браузер автоматически")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    {"check": cmd_check, "train": cmd_train, "plan": cmd_plan, "template": cmd_template,
     "table": cmd_table, "serve": cmd_serve}[args.cmd](cfg, args)


def cmd_serve(cfg, args):
    from .server import serve
    serve(cfg, port=args.port, open_browser=not args.no_browser)
