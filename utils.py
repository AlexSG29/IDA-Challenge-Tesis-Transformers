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
    """kind: 'operational_readouts', 'specifications', 'tte' o 'labels'."""
    return pd.read_csv(RAW / f"{split}_{kind}.csv")


def to_parquet(df: pd.DataFrame, name: str) -> Path:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    path = PROCESSED / f"{name}.parquet"
    df.to_parquet(path, index=False)
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
    """Convierte contadores e histogramas acumulados en incrementos por unidad de tiempo
    y agrega proporciones de bins para cada histograma.

    - d_<col>: (x_t - x_{t-1}) / Δt por vehículo; la primera lectura queda NaN.
    - p_<col>: proporción del bin dentro del histograma acumulado en t.
    """
    df = df.sort_values([ID, T]).reset_index(drop=True)
    counters, histograms = feature_groups(df)
    cum_cols = counters + [c for cols in histograms.values() for c in cols]

    g = df.groupby(ID, sort=False)
    dt = g[T].diff()
    out = {"dt": dt}
    diffs = g[cum_cols].diff()
    for c in cum_cols:
        out[f"d_{c}"] = diffs[c] / dt

    for k, cols in histograms.items():
        total = df[cols].sum(axis=1)
        for c in cols:
            out[f"p_{c}"] = df[c] / total.where(total > 0)

    return pd.concat([df, pd.DataFrame(out, index=df.index)], axis=1)
