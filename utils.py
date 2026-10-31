"""Funciones compartidas para la tesis (Scania Component X)."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize
from scipy.special import expit, log_expit

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


# ---------------------------------------------------------------- variables tabulares
def build_features(ro: pd.DataFrame, points: pd.DataFrame, n_last: int = 5,
                   chunk: int = 10) -> pd.DataFrame:
    """Una fila por punto de corte (vehicle_id, time_step) con el resumen de la historia hasta t*.

    - <var>_last: valor en t* (acumulados originales, tasas d_, proporciones p_ y dt).
    - <var>_mean y <var>_slope: media y pendiente respecto al tiempo en las últimas
      n_last lecturas (ignorando faltantes). Solo para d_, p_ y dt.
    - time_step: edad del vehículo en t*.
    ro debe venir ordenado por vehicle_id y time_step (como lo deja 01_datos).
    """
    vid, t = ro[ID].to_numpy(), ro[T].to_numpy(np.float64)
    ok = (np.diff(vid) > 0) | ((np.diff(vid) == 0) & (np.diff(t) > 0))
    assert ok.all(), "ro debe venir ordenado por vehicle_id y time_step"

    keys = pd.DataFrame({ID: vid, T: t, "_i": np.arange(len(ro))})
    pts = points[[ID, T]].drop_duplicates().astype({T: np.float64})
    idx = np.sort(keys.merge(pts, on=[ID, T])["_i"].to_numpy())
    del keys
    first = np.r_[True, vid[1:] != vid[:-1]]
    gstart = np.maximum.accumulate(np.where(first, np.arange(len(ro)), 0))
    start = np.maximum(idx - n_last + 1, gstart[idx])

    def wsum(a):  # suma de cada ventana [start, idx] con sumas acumuladas
        cs = np.concatenate([np.zeros((1, a.shape[1])), np.cumsum(a, axis=0)])
        return cs[idx + 1] - cs[start]

    raw = [c for c in ro.columns if c[0].isdigit()]
    dyn = [c for c in ro.columns if c.startswith(("d_", "p_"))] + ["dt"]
    blocks = [pd.DataFrame({ID: vid[idx], T: t[idx]}),
              pd.DataFrame(ro[raw].to_numpy(np.float32)[idx], columns=[f"{c}_last" for c in raw])]

    for i in range(0, len(dyn), chunk):  # por bloques para no agotar la RAM
        cols = dyn[i:i + chunk]
        x = ro[cols].to_numpy(np.float64)
        m = ~np.isnan(x)
        x0 = np.where(m, x, 0.0)
        tt = np.where(m, t[:, None], 0.0)
        n, sx, st = wsum(m.astype(np.float64)), wsum(x0), wsum(tt)
        sxt, stt = wsum(x0 * tt), wsum(tt * tt)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = sx / n
            var_t = stt / n - (st / n) ** 2
            slope = (sxt / n - (st / n) * mean) / var_t
        slope[(n < 2) | (var_t <= 1e-9)] = np.nan
        for name, arr in (("last", x[idx]), ("mean", mean), ("slope", slope)):
            blocks.append(pd.DataFrame(arr.astype(np.float32),
                                       columns=[f"{c}_{name}" for c in cols]))
    return pd.concat(blocks, axis=1)


# ---------------------------------------------------------------- costos y decisiones
# Fila = clase real, columna = clase predicha (IDA 2024 Industrial Challenge)
COST = np.array([[  0,   7,   8,   9,  10],
                 [200,   0,   7,   8,   9],
                 [300, 200,   0,   7,   8],
                 [400, 300, 200,   0,   7],
                 [500, 400, 300, 200,   0]])


def total_cost(y_true, y_pred, cost=COST) -> int:
    return int(cost[np.asarray(y_true, int), np.asarray(y_pred, int)].sum())


def decide_argmax(P: np.ndarray) -> np.ndarray:
    """Clase más probable."""
    return P.argmax(axis=1)


def decide_min_cost(P: np.ndarray, cost=COST) -> np.ndarray:
    """Clase con menor costo esperado: argmin_m sum_k p_k C[k, m]."""
    return (P @ cost).argmin(axis=1)


def calibration_split(vehicle_ids, frac: float = 0.2, seed: int = 0) -> set:
    """Conjunto de vehículos reservados para calibración (partición por vehículo)."""
    v = np.unique(vehicle_ids)
    rng = np.random.default_rng(seed)
    return set(rng.choice(v, size=int(round(frac * len(v))), replace=False))


# ---------------------------------------------------------------- modelos
def train_lgbm(X_fit, y_fit, X_val, y_val, params=None, seed=0, w_fit=None):
    """LightGBM multiclase con early stopping en validación. w_fit = pesos opcionales."""
    p = dict(objective="multiclass", num_class=5, learning_rate=0.05, num_leaves=31,
             min_data_in_leaf=50, feature_fraction=0.5, bagging_fraction=0.8,
             bagging_freq=1, seed=seed, verbose=-1)
    p.update(params or {})
    dfit = lgb.Dataset(X_fit, y_fit, weight=w_fit)
    dval = lgb.Dataset(X_val, y_val, reference=dfit)
    return lgb.train(p, dfit, num_boost_round=2000, valid_sets=[dval],
                     callbacks=[lgb.early_stopping(100, verbose=False)])

# ---------------------------------------------------------------- cambio de distribución
# Contadores cuya escala no cambia entre particiones (razón de medianas ≈ 1)
STABLE_COUNTERS = ["171_0", "835_0", "427_0"]
POS_COLS = [T, "n_hist"]  # posición del corte: edad y n.º de lecturas hasta t*


def is_invariant(col: str) -> bool:
    """True si la variable no depende del escalamiento por partición."""
    if col == T or col.startswith("Spec_"):
        return True                                   # edad y especificaciones
    if col.startswith(("p_", "dt_")):
        return True                                   # proporciones e intervalos
    return any(col in (f"{k}_last", f"d_{k}_last", f"d_{k}_mean", f"d_{k}_slope")
               for k in STABLE_COUNTERS)              # contadores estables


def invariant_features(cols) -> list[str]:
    """Filtra una lista de columnas y deja solo las invariantes (sin vehicle_id)."""
    return [c for c in cols if c != ID and is_invariant(c)]


def adversarial_auc(A: pd.DataFrame, B: pd.DataFrame, cols, w_A=None, seed: int = 0):
    """AUC de un clasificador que distingue filas de A (0) y de B (1).

    0,5 = indistinguibles; 1,0 = separables por completo. w_A = pesos opcionales para A.
    Devuelve (auc, importancia de cada variable).
    """
    X = pd.concat([A[cols], B[cols]], ignore_index=True)
    X = X.astype({c: "category" for c in cols if c.startswith("Spec_")})
    y = np.r_[np.zeros(len(A)), np.ones(len(B))]
    w = np.r_[np.ones(len(A)) if w_A is None else np.asarray(w_A, float), np.ones(len(B))]
    idx = np.random.default_rng(seed).permutation(len(X))
    tr, te = idx[: len(idx) // 2], idx[len(idx) // 2:]
    m = lgb.train(dict(objective="binary", verbose=-1, seed=seed),
                  lgb.Dataset(X.iloc[tr], y[tr], weight=w[tr]), num_boost_round=200)
    auc = roc_auc_score(y[te], m.predict(X.iloc[te]), sample_weight=w[te])
    imp = pd.Series(m.feature_importance("gain"), index=cols).sort_values(ascending=False)
    return auc, imp


def add_position(split: str, pts: pd.DataFrame) -> pd.DataFrame:
    """Agrega n_hist (n.º de lecturas hasta t*, inclusive) a cada punto de corte."""
    ro = pd.read_parquet(PROCESSED / f"{split}_readouts.parquet", columns=[ID, T])
    ro["n_hist"] = ro.groupby(ID).cumcount() + 1
    return pts[[ID, T]].merge(ro, on=[ID, T], how="left", validate="many_to_one")


def position_weights(train_pts: pd.DataFrame, target_pts: pd.DataFrame,
                     clip_q: float = 0.99, seed: int = 0) -> np.ndarray:
    """Pesos de importancia para que los cortes de train imiten la posición de target.

    w = P(target | edad, n_hist) / P(train | edad, n_hist), recortado en el cuantil clip_q
    y normalizado a media 1. Usa solo la posición de los cortes, nunca etiquetas.
    """
    X = pd.concat([train_pts[POS_COLS], target_pts[POS_COLS]], ignore_index=True)
    y = np.r_[np.zeros(len(train_pts)), np.ones(len(target_pts))]
    m = lgb.train(dict(objective="binary", num_leaves=8, min_data_in_leaf=200,
                       verbose=-1, seed=seed), lgb.Dataset(X, y), num_boost_round=100)
    p = np.clip(m.predict(train_pts[POS_COLS]), 1e-3, 1 - 1e-3)
    w = p / (1 - p)
    w = np.minimum(w, np.quantile(w, clip_q))
    return w / w.mean()


    # ---------------------------------------------------------------- predicciones guardadas
PREDS = PROCESSED / "preds"
PRED_COLS = [f"p{k}" for k in range(5)]


def save_predictions(model: str, form: str, split: str, df: pd.DataFrame,
                     P: np.ndarray, y=None, w=None) -> Path:
    """Guarda probabilidades por punto de corte: data/processed/preds/<model>_<form>_<split>.parquet"""
    PREDS.mkdir(parents=True, exist_ok=True)
    out = df[[ID, T]].reset_index(drop=True).copy()
    out[PRED_COLS] = P
    if y is not None:
        out["y"] = np.asarray(y, int)
    out["w"] = 1.0 if w is None else np.asarray(w, float)
    path = PREDS / f"{model}_{form}_{split}.parquet"
    out.to_parquet(path, index=False)
    return path


def load_predictions(model: str, form: str, split: str) -> pd.DataFrame:
    return pd.read_parquet(PREDS / f"{model}_{form}_{split}.parquet")


# ---------------------------------------------------------------- calibración
class TemperatureScaler:
    """Divide los log-probabilidades por una temperatura t que minimiza el log-loss."""
    def fit(self, P, y, w=None):
        y = np.asarray(y, int)
        w = np.ones(len(y)) if w is None else np.asarray(w, float)
        L = np.log(np.clip(P, 1e-12, 1))
        def nll(t):
            Z = L / t
            Z -= Z.max(1, keepdims=True)
            lp = Z - np.log(np.exp(Z).sum(1, keepdims=True))
            return -(w * lp[np.arange(len(y)), y]).sum() / w.sum()
        self.t_ = minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x
        return self

    def transform(self, P):
        Z = np.log(np.clip(P, 1e-12, 1)) / self.t_
        Z -= Z.max(1, keepdims=True)
        E = np.exp(Z)
        return E / E.sum(1, keepdims=True)


class IsotonicOVR:
    """Regresión isotónica por clase (uno contra el resto) y renormalización."""
    def fit(self, P, y, w=None):
        y = np.asarray(y, int)
        self.iso_ = [IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
                     .fit(P[:, k], (y == k).astype(float), sample_weight=w)
                     for k in range(P.shape[1])]
        return self

    def transform(self, P):
        Q = np.column_stack([iso.predict(P[:, k]) for k, iso in enumerate(self.iso_)]) + 1e-6
        return Q / Q.sum(1, keepdims=True)


def brier(y, P, w=None) -> float:
    """Puntaje de Brier multiclase (0 = perfecto)."""
    Y = np.eye(P.shape[1])[np.asarray(y, int)]
    return float(np.average(((P - Y) ** 2).sum(1), weights=w))


# ---------------------------------------------------------------- regla de umbral
def decide_threshold(P: np.ndarray, theta: float) -> np.ndarray:
    """Clase 4 si P(falla) = 1 - p0 >= theta; si no, clase 0."""
    return np.where(1 - P[:, 0] >= theta, 4, 0)


def best_threshold(P, y, grid=None, cost=COST):
    """Umbral que minimiza el costo total (se busca en validation)."""
    grid = np.linspace(0.001, 0.5, 500) if grid is None else grid
    costs = [total_cost(y, decide_threshold(P, t), cost) for t in grid]
    i = int(np.argmin(costs))
    return float(grid[i]), costs[i]


    # ---------------------------------------------------------------- incertidumbre
def paired_bootstrap(y, yhat_a, yhat_b, B: int = 2000, seed: int = 0, cost=COST) -> dict:
    """Diferencia de costo total (A - B) con IC percentil al 95 %, remuestreando vehículos."""
    y = np.asarray(y, int)
    ca = cost[y, np.asarray(yhat_a, int)]
    cb = cost[y, np.asarray(yhat_b, int)]
    diff = ca - cb
    rng = np.random.default_rng(seed)
    boots = np.array([diff[rng.integers(0, len(diff), len(diff))].sum() for _ in range(B)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"diferencia": int(diff.sum()), "IC95_inf": int(lo), "IC95_sup": int(hi),
            "significativa": bool(lo > 0 or hi < 0)}



# ---------------------------------------------------------------- riesgo en tiempo discreto
HAZ_EDGES = np.arange(0, 49, 6)          # 0, 6, ..., 48 -> 8 intervalos de amplitud 6
CLASS_INTERVALS = {4: [0], 3: [1], 2: [2, 3], 1: [4, 5, 6, 7]}   # intervalos de cada clase


def hazard_targets(time_left, event, censor_as_survival: bool = False):
    """Matrices persona-periodo (n x 8): R = en riesgo en el intervalo j, E = falla en j.

    - Falla en t <= 48: en riesgo hasta su intervalo, con E = 1 en ese intervalo.
    - Falla o censura con t > 48: sobrevive los 8 intervalos.
    - Censurado con t <= 48: solo los intervalos sobrevividos completos.
      Con censor_as_survival=True (formulación convencional) se asume que sobrevive los 8.
    """
    t = np.asarray(time_left, float)
    e = np.asarray(event, int)
    lo, hi = HAZ_EDGES[:-1], HAZ_EDGES[1:]
    j_ev = np.searchsorted(hi, t, side="left")
    R = np.zeros((len(t), len(lo)), np.float32)
    E = np.zeros_like(R)
    ev = (e == 1) & (t <= HAZ_EDGES[-1])
    surv = (t > HAZ_EDGES[-1]) | ((e == 0) & censor_as_survival)
    R[surv] = 1
    for i in np.where(ev)[0]:
        R[i, : j_ev[i] + 1] = 1
        E[i, j_ev[i]] = 1
    cens = (e == 0) & ~surv
    R[cens] = hi[None, :] <= t[cens, None]
    return R, E


class DiscreteHazard:
    """Regresión logística de riesgo discreto: logit h_j(x) = a_j + x·b, con penalización L2."""
    def __init__(self, l2: float = 1.0, max_iter: int = 300):
        self.l2, self.max_iter = l2, max_iter

    def fit(self, X, R, E, w=None):
        X = np.asarray(X, np.float32)
        n, d = X.shape
        J = R.shape[1]
        w = np.ones(n) if w is None else np.asarray(w, float)
        WR = (w[:, None] * R).astype(np.float64)
        sw = WR.sum()

        def f(theta):
            a, b = theta[:J], theta[J:]
            Z = (X @ b.astype(np.float32)).astype(np.float64)[:, None] + a[None, :]
            ll = E * log_expit(Z) + (1 - E) * log_expit(-Z)
            loss = -(WR * ll).sum() / sw + 0.5 * self.l2 * (b @ b) / n
            G = WR * (expit(Z) - E) / sw
            gb = (X.T @ G.sum(1).astype(np.float32)).astype(np.float64) + self.l2 * b / n
            return loss, np.r_[G.sum(0), gb]

        p0 = np.clip((WR * E).sum(0) / WR.sum(0), 1e-4, 0.5)
        th0 = np.r_[np.log(p0 / (1 - p0)), np.zeros(d)]
        self.res_ = minimize(f, th0, jac=True, method="L-BFGS-B",
                             options={"maxiter": self.max_iter})
        self.a_, self.b_ = self.res_.x[:J], self.res_.x[J:]
        return self

    def hazards(self, X):
        Z = (np.asarray(X, np.float32) @ self.b_.astype(np.float32))[:, None] + self.a_[None, :]
        return expit(Z.astype(np.float64))

    def predict_proba(self, X):
        """Probabilidades de las 5 clases: P(falla en I_j) = h_j · S_{j-1}; clase 0 = S_8."""
        h = self.hazards(X)
        S_prev = np.cumprod(np.c_[np.ones(len(h)), 1 - h[:, :-1]], axis=1)
        Pint = h * S_prev
        P = np.zeros((len(h), 5))
        P[:, 0] = np.prod(1 - h, axis=1)
        for k, js in CLASS_INTERVALS.items():
            P[:, k] = Pint[:, js].sum(1)
        return P


class TabularPreprocessor:
    """Para modelos lineales: mediana para faltantes, estandarización y one-hot de Spec_*.
    Se ajusta solo con datos de entrenamiento; categorías nuevas quedan en cero."""
    def fit(self, df, cols):
        self.cat_ = [c for c in cols if c.startswith("Spec_")]
        self.num_ = [c for c in cols if c not in self.cat_]
        x = df[self.num_].astype(np.float64)
        self.med_ = x.median()
        x = x.fillna(self.med_)
        self.mu_, self.sd_ = x.mean(), x.std().replace(0, 1).fillna(1)
        self.levels_ = {c: sorted(df[c].astype(str).unique()) for c in self.cat_}
        return self

    def transform(self, df):
        x = ((df[self.num_].astype(np.float64).fillna(self.med_) - self.mu_) / self.sd_).clip(-10, 10)
        parts = [x.to_numpy(np.float32)]
        for c in self.cat_:
            s = df[c].astype(str).to_numpy()
            parts.append(np.column_stack([s == l for l in self.levels_[c][1:]]).astype(np.float32))
        return np.hstack(parts)


# ---------------------------------------------------------------- regresión logística ordinal
class OrdinalLogistic:
    """Odds proporcionales: P(Y <= k | x) = σ(θ_k − x·b), k = 0..3, con θ crecientes y L2 en b."""
    def __init__(self, n_classes: int = 5, l2: float = 1.0, max_iter: int = 300):
        self.K, self.l2, self.max_iter = n_classes, l2, max_iter

    def _theta(self, t):
        return np.cumsum(np.r_[t[0], np.exp(t[1:])])   # garantiza θ_0 < θ_1 < ... < θ_3

    def _f(self, params, X, y, w):
        K, n = self.K, X.shape[0]
        t, b = params[:K - 1], params[K - 1:]
        th = self._theta(t)
        eta = (X @ b.astype(np.float32)).astype(np.float64)
        su = expit(np.r_[th, np.inf][y] - eta)          # P(Y <= y)
        sl = expit(np.r_[-np.inf, th][y] - eta)         # P(Y <= y-1)
        p = np.clip(su - sl, 1e-12, None)
        sw = w.sum()
        loss = -(w * np.log(p)).sum() / sw + 0.5 * self.l2 * (b @ b) / n
        du, dl = su * (1 - su), sl * (1 - sl)
        g_eta = w * (du - dl) / p / sw
        gb = (X.T @ g_eta.astype(np.float32)).astype(np.float64) + self.l2 * b / n
        g_th = np.zeros(K - 1)
        m = y < K - 1
        np.add.at(g_th, y[m], -(w * du / p)[m] / sw)
        m = y > 0
        np.add.at(g_th, y[m] - 1, (w * dl / p)[m] / sw)
        g_t = np.zeros(K - 1)
        g_t[0] = g_th.sum()
        for j in range(1, K - 1):
            g_t[j] = np.exp(t[j]) * g_th[j:].sum()
        return loss, np.r_[g_t, gb]

    def fit(self, X, y, w=None):
        X = np.asarray(X, np.float32)
        y = np.asarray(y, int)
        w = np.ones(len(y)) if w is None else np.asarray(w, float)
        cum = np.cumsum(np.bincount(y, weights=w, minlength=self.K))[:-1] / w.sum()
        cum = np.clip(cum, 1e-4, 1 - 1e-4)
        th0 = np.log(cum / (1 - cum))
        t0 = np.r_[th0[0], np.log(np.maximum(np.diff(th0), 1e-3))]
        self.res_ = minimize(self._f, np.r_[t0, np.zeros(X.shape[1])], args=(X, y, w),
                             jac=True, method="L-BFGS-B", options={"maxiter": self.max_iter})
        self.t_, self.b_ = self.res_.x[:self.K - 1], self.res_.x[self.K - 1:]
        return self

    def predict_proba(self, X):
        th = self._theta(self.t_)
        eta = (np.asarray(X, np.float32) @ self.b_.astype(np.float32)).astype(np.float64)
        C = expit(th[None, :] - eta[:, None])
        C = np.c_[np.zeros(len(eta)), C, np.ones(len(eta))]
        return np.clip(np.diff(C, axis=1), 1e-12, 1)