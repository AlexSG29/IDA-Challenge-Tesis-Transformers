"""Funciones compartidas para la tesis (Scania Component X)."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

SPLITS = ("train", "validation", "test")
ID, T = "vehicle_id", "time_step"


# ---------------------------------------------------------------- carga
def load_raw(split: str, kind: str) -> pd.DataFrame:
    """kind: 'operational_readouts', 'specifications', 'tte' o 'labels'.

    Busca en data/raw/<split>/ y, si no está, en data/raw/.
    """
    name = f"{split}_{kind}.csv"
    for path in (RAW / split / name, RAW / name):
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(f"No encontré {name} en {RAW / split} ni en {RAW}")


def to_parquet(df: pd.DataFrame, name: str) -> Path:
    """Guarda en data/processed/ con floats en float32 y compresión zstd."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    floats = df.select_dtypes("float64").columns
    df = df.astype({c: "float32" for c in floats})
    path = PROCESSED / f"{name}.parquet"
    df.to_parquet(path, index=False, compression="zstd")
    return path


def load_processed(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / f"{name}.parquet")


# ---------------------------------------------------- grupos de variables
def feature_groups(df: pd.DataFrame) -> tuple[list[str], dict[str, list[str]]]:
    """Separa contadores (un solo bin) e histogramas (varios bins por prefijo).

    Las columnas siguen el patrón '<id>_<bin>' (p. ej. '171_0', '167_0'...'167_9').
    """
    groups: dict[str, list[str]] = {}
    for c in df.columns:
        m = re.fullmatch(r"(\d+)_(\d+)", c)
        if m:
            groups.setdefault(m.group(1), []).append(c)
    for k in groups:
        groups[k].sort(key=lambda c: int(c.split("_")[1]))
    counters = [cols[0] for cols in groups.values() if len(cols) == 1]
    histograms = {k: cols for k, cols in groups.items() if len(cols) > 1}
    return counters, histograms


# --------------------------------------------- acumulados -> incrementos
def add_increments(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte contadores e histogramas acumulados en tasas por unidad de tiempo.

    - d_<col>: (x_t - x_última_válida) / (t - t_última_válida), por vehículo.
      Si falta la lectura actual, o el valor disminuye (error de registro), queda NaN.
    - p_<col>: proporción de cada bin dentro de su histograma acumulado en t.
    - dt: tiempo desde la lectura anterior.
    """
    df = df.sort_values([ID, T]).reset_index(drop=True)
    counters, histograms = feature_groups(df)
    blocks = {c: [c] for c in counters} | histograms
    out = {}

    for cols in blocks.values():
        valid = df[cols].notna().all(axis=1)
        t_last = df[T].where(valid).groupby(df[ID]).shift().groupby(df[ID]).ffill()
        x_last = df[cols].where(valid, axis=0).groupby(df[ID]).shift().groupby(df[ID]).ffill()
        inc = df[cols].sub(x_last).div((df[T] - t_last).where(valid), axis=0)
        inc = inc.mask(inc < 0)
        for c in cols:
            out[f"d_{c}"] = inc[c]

        if len(cols) > 1:
            total = df[cols].sum(axis=1).where(valid)
            for c in cols:
                out[f"p_{c}"] = df[c] / total.where(total > 0)

    out["dt"] = df.groupby(ID)[T].diff()
    return pd.concat([df, pd.DataFrame(out, index=df.index)], axis=1)