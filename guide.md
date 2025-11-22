✅ METODOLOGÍA PROPUESTA (GUÍA PASO A PASO)

Objetivo: entrenar un Transformer para predecir la clase 0–4 para cada vehículo (en su último readout), minimizar el costo total y preparar el archivo de predicciones.

🔵 FASE 1 — ENTENDER Y PREPARAR LOS DATOS (lo primero que harás)
1.1 Cargar archivos grandes de lectura operacional

train_operational_readouts.csv

validation_operational_readouts.csv

test_operational_readouts.csv

Cada archivo tiene:

vehicle_id

time_step

~14 variables (algunas con bins)

👉 Son secuencias multivariadas irregulares → IDEAL para Transformers.

1.2 Cargar la información complementaria

train_specifications.csv

validation_specifications.csv

test_specifications.csv

👉 Uso: codificación categórica (embeddings).

1.3 Cargar los labels

train_tte.csv → se usa solo para construir etiquetas 0–4

validation_labels.csv → etiquetas ya listas

test → no tiene labels

1.4 Construir la etiqueta (clase 0–4) para el train

Según las reglas:

Clase	Tiempo antes de falla
1	48–24
2	24–12
3	12–6
4	6–0
0	No falló

👉 Procesamiento:
Para cada vehicle_id del training:

Tomar su length_of_study_time_step desde train_tte.csv

Recorrer sus readouts.

Según el tiempo relativo al evento → asignar clase 0–4 al último readout.

Esto produce tu dataset final:

vehicle_id | sequence_of_readouts | specs | label

🔵 FASE 2 — CREAR EL DATASET FINAL LISTO PARA TRANSFORMERS
2.1 Agrupar por vehículo

Para cada vehículo:

ordenar por time_step

normalizar cada feature

secuencia = matriz [longitud variable x n_features]

Ejemplo:

Vehículo 123  
[[0.21, 1, 0, ...],
 [0.23, 0, 1, ...],
 [0.19, 1, 0, ...], ...]

2.2 Cortar secuencias o aplicar padding

Transformers requieren un tamaño fijo → recomiendo:

MAX_SEQ_LEN ≈ 200–300

Padding + máscara de atención

2.3 Procesar especificaciones

Usar embeddings tipo NLP:

engine_type → embedding 16d  
wheel_config → embedding 8d  


Y concatenar al embedding del CLS token.

🔵 FASE 3 — MODELADO (Transformers)

Aquí empieza lo fuerte pero ya todo está organizado.

3.1 Arquitectura

Modelo recomendado:

[CLS] + embeddings de secuencia → Transformer Encoder → MLP → clase 0–4


Componentes:

Embedding denso para features numéricos

Positional encoding aprendible

2–3 capas Transformer Encoder

Dropout bajo (0.1)

Capa final de clasificación con 5 logits

3.2 Función de pérdida

Este challenge usa COSTOS → NO accuracy.

Debes usar Cost-sensitive loss:

loss = cross_entropy * weight[label][pred]


O usar matriz de costo directamente.

Esto es la “gracia” del challenge:
👉 No gana quien tiene mejor accuracy, sino quien minimiza el costo.

3.3 Entrenamiento

batch size = 16–32 (dependiendo de GPU)

10–20 epochs

early stopping por costo validación

optimizador AdamW

🔵 FASE 4 — VALIDACIÓN

Usa:

Secuencias truncadas de validación

Labels de validation_labels.csv

Evalúa:

matriz de confusión

Total_cost usando la matriz del challenge

Si el costo es muy alto, ajustar:

class weights

número de capas

embeddings

tamaño secuencia

🔵 FASE 5 — GENERAR PREDICCIONES DEL TEST

El archivo de salida debe tener:

vehicle_id, predicted_class


Pasos:

Cargar secuencias completas del test.

Tomar solo el último readout por vehículo (ya está así el dataset).

Preprocesar igual que train/val.

Pasar por el modelo.

Generar clase 0–4.

Exportar como IDA_Industrial_challenge_2024_predictions.csv.

🔵 FASE 6 — RESULTADOS

Tu paper debe mostrar:

Gráficos de distribución de clases

Arquitectura del Transformer

Tabla comparativa: LSTM vs Transformer

Matriz de costos obtenida

Ablation de hiperparámetros

Justificación:
“Los Transformers capturan dependencias globales entre sensores multivariados de forma efectiva”.

🔵 FASE 7 — ENTREGA FINAL

Enviar:

IDA_Industrial_challenge_2024_predictions.csv

Paper en formato de la conferencia

✅ RESUMEN SUPER CORTO PARA TI (tu checklist real):
Paso 1 — Preparar etiquetas del train

Usar train_tte + reglas 48–0.

Paso 2 — Construir secuencias por vehículo

Agrupar, ordenar, pad, normalizar.

Paso 3 — Embeddings para especificaciones
Paso 4 — Entrenar Transformer + cost-sensitive loss
Paso 5 — Validar usando validation_labels
Paso 6 — Predecir test
Paso 7 — Generar CSV
