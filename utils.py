"""Funciones compartidas para la tesis (Scania Component X)."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize
from scipy.special import expit, log_expit

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

ID, T = "vehicle_id", "time_step"

