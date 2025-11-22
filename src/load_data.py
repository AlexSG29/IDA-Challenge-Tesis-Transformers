# Vamos a cargar los datos desde los archivos CSV y prepararlos para su análisis.

import pandas as pd

""" 
El primer dato sea el de "train_operational_readouts.csv" 
lo haremos en forma de función porque cuando llamemos a esta función
desde otro archivo, nos va a permitir reutilizar el código. 
"""

TOR = 'data/raw/train_operational_readouts.csv'

def data (file_path):
    df = pd.read_csv(file_path)
    return df

# hacemos una prueba para ver si la función funciona correctamente


print(data(TOR).head())
