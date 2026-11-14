#!/usr/bin/env bash
# Запуск на macOS / Linux:  bash запуск.sh   (на macOS можно дважды щёлкнуть «Запуск.command»)
set -e
cd "$(dirname "$0")"

if [ ! -f .venv/installed.ok ]; then
  PY=""
  for c in python3 python; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
      PY="$c"; break
    fi
  done
  if [ -z "$PY" ]; then
    echo "Нужен Python 3.11 или новее: https://www.python.org/downloads/"
    exit 1
  fi
  echo "Первый запуск: установка компонентов (1-3 минуты, нужен интернет)..."
  [ -x .venv/bin/python ] || "$PY" -m venv .venv
  .venv/bin/python -m pip install --disable-pip-version-check -q -r requirements.txt
  touch .venv/installed.ok
fi

exec .venv/bin/python -m nomogram serve
