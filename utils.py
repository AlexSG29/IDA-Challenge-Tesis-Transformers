"""Funciones compartidas para la tesis (Scania Component X)."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import numpy as np

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


    # ---------------------------------------------------------------- etiquetas
HORIZON = 48  # ventana máxima de las clases 1–4


def tte_to_class(time_left: pd.Series) -> pd.Series:
    """Clase del challenge: 4 = (0,6], 3 = (6,12], 2 = (12,24], 1 = (24,48], 0 = >48."""
    bins = [-np.inf, 6, 12, 24, 48, np.inf]
    return pd.cut(time_left, bins=bins, labels=[4, 3, 2, 1, 0], right=True).astype(int)


def build_landmarks(ro: pd.DataFrame, tte: pd.DataFrame, k: int = 10,
                    min_history: int = 5, seed: int = 0) -> pd.DataFrame:
    """Elige al azar hasta k lecturas por vehículo como puntos de predicción t*.

    - time_left: tiempo desde t* hasta la reparación o el fin del estudio.
    
    - event: 1 si hubo reparación, 0 si está censurado.
    - y_conv: clase convencional (los censurados se etiquetan 0).
    - y_cens: clase con censura; NaN si está censurado y quedan ≤ 48 unidades.
    """
    rng = np.random.default_rng(seed)
    lm = ro[[ID, T]].sort_values([ID, T]).copy()
    lm["n_hist"] = lm.groupby(ID).cumcount() + 1
    lm = lm[lm["n_hist"] >= min_history]
    lm["_r"] = rng.random(len(lm))
    lm = lm[lm.groupby(ID)["_r"].rank(method="first") <= k].drop(columns="_r")

    lm = lm.merge(tte, on=ID)
    lm["time_left"] = lm["length_of_study_time_step"] - lm[T]
    lm["event"] = lm["in_study_repair"]
    lm["y_conv"] = tte_to_class(lm["time_left"]).where(lm["event"] == 1, 0)
    known = (lm["event"] == 1) | (lm["time_left"] > HORIZON)
    lm["y_cens"] = lm["y_conv"].where(known).astype("Int64")
    return (lm.drop(columns=["length_of_study_time_step", "in_study_repair"])
              .sort_values([ID, T]).reset_index(drop=True))


def eval_points(ro: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Última lectura de cada vehículo + su class_label (validation y test)."""
    last = ro[[ID, T]].sort_values([ID, T]).groupby(ID).tail(1)
    return last.merge(labels, on=ID, validate="one_to_one").reset_index(drop=True)