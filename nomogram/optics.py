"""Оптические расчёты: векторы силы (Thibos), роговичный астигматизм, глубина абляции."""
import numpy as np


def _arr(x):
    return np.asarray(x, dtype=float)


def to_vectors(sph, cyl, axis):
    """Сфера/цилиндр/ось -> (M, J0, J45). Не зависит от записи цилиндра со знаком + или −."""
    sph, cyl, axis = _arr(sph), _arr(cyl), _arr(axis)
    axis = np.where(np.isnan(axis) & (cyl == 0), 0.0, axis)
    a = np.deg2rad(2 * axis)
    m = sph + cyl / 2
    j0 = -cyl / 2 * np.cos(a)
    j45 = -cyl / 2 * np.sin(a)
    return m, j0, j45


def from_vectors(m, j0, j45, minus_cyl=True):
    """(M, J0, J45) -> сфера, цилиндр, ось (ось 1..180)."""
    m, j0, j45 = _arr(m), _arr(j0), _arr(j45)
    cyl = -2 * np.hypot(j0, j45)
    axis = (np.rad2deg(np.arctan2(j45, j0)) / 2) % 180
    sph = m - cyl / 2
    if not minus_cyl:
        sph, cyl, axis = sph + cyl, -cyl, (axis + 90) % 180
    axis = np.where(np.round(axis) == 0, 180.0, np.round(axis))
    return sph, cyl, axis


def corneal_vectors(k1, k1_axis, k2, k2_axis):
    """Роговичный астигматизм по кератометрии в виде (J0, J45) в рефракционных обозначениях."""
    k1, k2, k1_axis, k2_axis = _arr(k1), _arr(k2), _arr(k1_axis), _arr(k2_axis)
    flat_axis = np.where(k1 <= k2, k1_axis, k2_axis)
    _, j0, j45 = to_vectors(0.0, -np.abs(k2 - k1), flat_axis)
    return j0, j45


def corneal_astigmatism(k1, k1_axis, k2, k2_axis):
    """Роговичный астигматизм одного глаза: (величина D, плоская ось °) или (nan, nan), если данных нет.

    Плоская ось соответствует оси минусового цилиндра в рефракции (прямой астигматизм — около 180°).
    """
    k1, k2, a1, a2 = (float(_arr(v)) for v in (k1, k2, k1_axis, k2_axis))
    if np.isnan(k1) or np.isnan(k2):
        return np.nan, np.nan
    flat, steep = (a1, a2) if k1 <= k2 else (a2, a1)
    if np.isnan(flat) and not np.isnan(steep):
        flat = steep + 90
    if np.isnan(flat):
        return abs(k2 - k1), np.nan
    return abs(k2 - k1), flat % 180


def axis_diff(a, b):
    """Знаковая разница осей b − a в пределах [−90; 90)."""
    return (float(b) - float(a) + 90) % 180 - 90


def ablation_depth_um(sph, cyl, oz):
    """Ориентировочная центральная глубина миопической абляции по Мюнерлину, мкм.

    Реальная глубина EX500 (Custom-Q/WFO, переходная зона) больше, итог сверять с лазером.
    Для гиперметропии центральная абляция ~0, возвращается 0.
    """
    sph, cyl, oz = _arr(sph), _arr(cyl), _arr(oz)
    strongest_myopic = np.maximum(-np.minimum(sph, sph + cyl), 0.0)
    return strongest_myopic * oz ** 2 / 3
