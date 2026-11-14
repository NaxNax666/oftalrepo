import tomllib
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.toml"


class Config(dict):
    """Словарь настроек + корневая папка проекта (относительно неё задаются пути)."""

    def __init__(self, data, root):
        super().__init__(data)
        self.root = Path(root)

    @property
    def models_dir(self):
        return self.root / "модели"

    @property
    def results_dir(self):
        return self.root / "результаты"


def load_config(path=None):
    path = Path(path or DEFAULT_CONFIG).resolve()
    with open(path, "rb") as f:
        return Config(tomllib.load(f), path.parent)
