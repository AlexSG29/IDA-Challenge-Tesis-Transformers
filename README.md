# Mantenimiento predictivo sensible al costo en vehículos pesados

Código de la tesis de maestría (Estadística Aplicada y Ciencia de Datos, UTB):
*modelos de tiempo hasta la falla con datos censurados e irregulares* sobre el
dataset **Scania Component X** (IDA 2024 Industrial Challenge).

## Estructura

```
data/raw/         CSV originales (no versionados)
data/processed/   Parquet derivados (no versionados)
notebooks/
  01_datos.ipynb        carga, incrementos de contadores, proporciones de histogramas
  02_etiquetas.ipynb    formulación convencional vs. censura + landmarking
  03_modelos.ipynb      ordinal, riesgo discreto, gradient boosting, Transformer
  04_evaluacion.ipynb   reglas de decisión, costo total, bootstrap pareado
utils.py          funciones compartidas
results/          tablas y figuras finales
```

## Datos

Descargar el dataset desde el repositorio de Scania / SND
(Kharazian et al. 2025, *Scientific Data*, doi:10.1038/s41597-025-04802-6)
y copiar los 9 CSV en `data/raw/`:

- `train_operational_readouts.csv`, `train_specifications.csv`, `train_tte.csv`
- `validation_operational_readouts.csv`, `validation_specifications.csv`, `validation_labels.csv`
- `test_operational_readouts.csv`, `test_specifications.csv`, `test_labels.csv`

## Entorno

```bash
uv sync                 # crea .venv con las dependencias
uv run nbstripout --install   # limpia salidas de notebooks en cada commit
```
