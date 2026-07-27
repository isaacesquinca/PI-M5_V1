# Predicción de Pago a Tiempo — PI-M5

## Caso de negocio

Una entidad de crédito necesita anticipar, al momento de originar un préstamo,
si un cliente **pagará a tiempo** (`Pago_atiempo`) o no. Predecirlo con
anticipación permite ajustar condiciones de crédito, priorizar seguimiento de
cartera y reducir pérdidas por mora, en vez de reaccionar después de que el
atraso ya ocurrió.

El proyecto cubre el ciclo completo: limpieza de datos, modelado supervisado
y no supervisado, y monitoreo del modelo en producción ante cambios en la
población de clientes (data drift).

---

## Estructura del proyecto

```
├── src/
│   ├── cargar_datos.py              # Carga el Excel crudo y el dataset ya limpio
│   ├── ft_engineering.py            # Preprocesamiento (imputación, one-hot, split)
│   ├── compresion_eda.ipynb         # Limpieza y EDA (crudo -> dataset_limpio.xlsx)
│   ├── model_training_evaluation.ipynb  # Modelos supervisados y no supervisados
│   ├── model_monitoring.py          # App de Streamlit: monitoreo y data drift
│   ├── exportar_modelo.py           # Entrena el mejor modelo y genera models/model.pkl
│   ├── model_deploy.py              # API FastAPI: expone el modelo en /predict
│   └── data/processed/dataset_limpio.xlsx
├── models/
│   └── model.pkl                    # Preprocesador + modelo, generado por exportar_modelo.py
├── Dockerfile
├── .dockerignore
└── requirements.txt
```

---

## 1. Datos y limpieza (`compresion_eda.ipynb`)

Dataset original: `Base_de_datos.xlsx`, 27 variables de un préstamo (monto,
plazo, edad, salario, historial en centrales de riesgo, etc.) más el target
`Pago_atiempo`.

Principales pasos de limpieza:
- Se eliminó `huella_consulta` (identificador sin valor predictivo).
- Se corrigieron outliers claramente atribuibles a error de captura
  (`tipo_credito=68`, `edad_cliente` en 121-123, `plazo_meses=90`).
- Se eliminaron ~24 registros con `salario_cliente=0` y filas con nulos en
  columnas clave (`puntaje_datacredito`, `saldo_mora`, `saldo_total`).
- Se imputó con mediana en `saldo_principal`, `saldo_mora_codeudor` y
  `promedio_ingresos_datacredito`.
- `tipo_credito` se recodificó de número a categoría (A-E).
- `fecha_prestamo` se descompuso en `anio_prestamo`, `mes_prestamo`,
  `trimestre`, `dia_mes`, `dia_semana` y `fin_de_semana`, y se eliminó la
  fecha cruda.

### Hallazgo clave: fuga de datos en `puntaje`

Al entrenar los primeros modelos, **todos** (Random Forest, XGBoost,
LightGBM, CatBoost, Stacking) daban accuracy/F1 de 1.0 — señal de que algo
andaba mal, no de un modelo perfecto. La variable `puntaje` correlacionaba
0.92 con el target y separaba las clases sin solapamiento (máximo de
`puntaje` en `Pago_atiempo=0` era 62.7; el mínimo en `Pago_atiempo=1` era
63.8). Todo indica que es un score que se actualiza *después* de conocerse el
resultado del pago, no algo disponible al momento de originar el crédito.
**Se excluyó del dataset** antes de entrenar cualquier modelo.

---

## 2. Preprocesamiento (`ft_engineering.py`)

`preprocesar_datos()` carga el dataset limpio y arma:
- Imputación por mediana (numéricas) y por moda + one-hot encoding
  (categóricas), vía un `ColumnTransformer`.
- `train_test_split` estratificado por `Pago_atiempo` (80/20).
- Una clave de orden temporal (`anio_prestamo`/`mes_prestamo`/`dia_mes`)
  para poder validar con `TimeSeriesSplit` más adelante, sin necesitar una
  fecha cruda.

Devuelve `X_train`, `X_test`, `y_train`, `y_test`, el `preprocessor` ya
ajustado y los nombres de las variables generadas.

---

## 3. Modelado supervisado (`model_training_evaluation.ipynb`)

Se entrenan y comparan:

| Modelo | Notas |
|---|---|
| Random Forest | Baseline |
| XGBoost | Con `GridSearchCV` (n_estimators, max_depth, learning_rate) |
| LightGBM | |
| CatBoost | |
| Stacking | Los 4 anteriores como base, regresión logística como meta-modelo |

Para cada modelo se reportan: matriz de confusión, accuracy, precision,
recall, F1, ROC-AUC, curvas ROC/Precision-Recall, y el top 10 de variables
más importantes. La validación cruzada se corre con `KFold`,
`StratifiedKFold` (ambas sobre `X_train` tal cual) y `TimeSeriesSplit`
(sobre `X_train` reordenado cronológicamente).

### Resultados (dataset sin `puntaje`)

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 0.9542 | 0.9541 | 1.0000 | 0.9765 | 0.6562 |
| XGBoost | 0.9542 | 0.9550 | 0.9990 | 0.9765 | 0.7018 |
| LightGBM | 0.9523 | 0.9549 | 0.9970 | 0.9755 | 0.6814 |
| CatBoost | 0.9537 | 0.9550 | 0.9985 | 0.9762 | 0.7063 |
| Stacking | 0.9542 | 0.9554 | 0.9985 | 0.9765 | 0.6952 |

Sin la fuga de datos, las métricas ya no son perfectas y son consistentes
entre validación cruzada (F1 ≈ 0.976 en KFold, StratifiedKFold y
TimeSeriesSplit para los 5 modelos) y test.

**Mejor modelo por F1: Random Forest** (empatado en la práctica con los
demás — la diferencia real está en ROC-AUC, donde CatBoost (0.706) y
XGBoost (0.702) separan mejor las clases que Random Forest (0.656)).

El recall cercano a 1.0 en todos los modelos, junto con un accuracy que
apenas supera el 95%, refleja el desbalance de clases (~95% de los
clientes sí paga a tiempo): los modelos detectan muy bien a quien paga,
pero el ROC-AUC moderado (0.66-0.71) muestra que separar a quien **no** va
a pagar sigue siendo el reto real — vale la pena mirar precision/recall de
la clase minoritaria antes de poner el modelo en producción, no solo el
accuracy global.

---

## 4. Modelado no supervisado — Segmentación de clientes

Sobre las mismas features preprocesadas (escaladas aparte con
`StandardScaler`, ya que K-Means/PCA sí son sensibles a la escala):

- **K-Means**: método del codo + Silhouette para elegir K entre 3, 4 y 5.
  K óptimo: **3** (Silhouette 0.1108, contra 0.1026 en K=4 y 0.1094 en K=5).
  Un Silhouette tan bajo (<0.25) indica una estructura de segmentación
  **débil**: los 3 clusters son útiles operacionalmente, pero los clientes
  no forman grupos naturalmente separados — las diferencias entre ellos
  son graduales, no categóricas.
- **DBSCAN** (`eps=0.5`, `min_samples=5`): con esos parámetros **no
  encontró ningún cluster denso** — el 100% de los puntos quedó marcado
  como ruido/outlier. Esto refuerza la lectura del Silhouette: los datos
  están relativamente distribuidos de forma homogénea, sin agrupaciones
  densas naturales. Si se quiere que DBSCAN sea útil aquí, habría que
  probar valores de `eps`/`min_samples` distintos, ajustados a la densidad
  real de este dataset.
- **PCA**: los 2 primeros componentes no alcanzan a resumir bien los datos
  — se necesitan 17 componentes para 80% de varianza y 20 para 90%,
  señal de que las variables aportan información de forma distribuida, sin
  unas pocas features dominantes.
- **t-SNE**: proyección 2D no lineal para visualizar los clusters de
  K-Means.
- **Tasa de incumplimiento por segmento**: 5.4% (cluster 0), 5.3% (cluster
  1) y 4.1% (cluster 2) — una diferencia real pero modesta entre
  segmentos, consistente con la estructura débil que ya señalaba el
  Silhouette.
- **Perfiles por cluster**: el cluster 0 destaca por mayor `salario_cliente`
  promedio (~$12.7M vs ~$3.5M en el cluster 1) y plazos más cortos
  (~8.9 meses vs ~12.5); aun así, no es el segmento con menor
  incumplimiento — el cluster 2 (plazos e ingresos intermedios) es el de
  menor riesgo.
- `cluster_risk_rank`: feature derivada que ordena los clusters por riesgo
  (0=cluster 0, mayor riesgo; 2=cluster 2, menor riesgo), disponible para
  agregarse a los modelos supervisados si mejora el desempeño.

`puntaje` se excluyó también aquí: incluirla agruparía a los clientes por el
propio resultado (pagó o no), no por sus características reales.

---

## 5. Monitoreo y Data Drift (`model_monitoring.py`)

App de Streamlit que compara la base con la que se entrenó el modelo
(**referencia**, `dataset_limpio.xlsx`) contra una base nueva
(**actual**, subida por el usuario) para detectar cambios en la población.

La base actual se sube **cruda** y la app la pasa por la misma limpieza de
`compresion_eda.ipynb` (corrección de outliers conocidos, eliminación de
filas con salario/nulos según las mismas reglas, imputación por mediana,
recategorización de `tendencia_ingresos`, mapeo de `tipo_credito` a A-E,
ingeniería de fecha, exclusión de `puntaje`) antes de compararla — así la
comparación de drift es **limpio vs. limpio**, en el mismo espacio de
variables que usa el modelo.

Si algún código de `tipo_credito` no está en el mapeo original (4, 6, 7, 9,
10), se redondea al código válido más cercano antes de mapear a letra —
esto permite seguir comparando esa variable como categórica limpia (no
"Desconocido") aunque los códigos se hayan corrido, y el drift real queda
visible en cómo cambia la *proporción* entre letras, no oculto. La app
muestra además un resumen expandible con cuántas filas/valores tocó cada
regla de limpieza al aplicarse sobre la base nueva, incluyendo cuántos
`tipo_credito` tuvieron que redondearse.

Métricas calculadas por variable:
- **KS test** y **PSI** (numéricas)
- **Jensen-Shannon divergence** (numéricas)
- **Chi-cuadrado** (categóricas)

Umbrales usados para marcar drift: PSI > 0.2 o KS con p-valor < 0.05
(numéricas); Chi-cuadrado con p-valor < 0.05 (categóricas).

La app también:
- Permite filtrar la base actual por rango de fechas (muestreo periódico),
  aplicado sobre la fecha cruda antes de limpiar (la fecha desaparece
  durante la limpieza).
- Entrena el modelo (Random Forest) y genera una tabla con los datos
  actuales ya limpios junto a sus predicciones y probabilidades.
- Si la base actual trae el target real, calcula el desempeño real del
  modelo sobre ella (no solo drift de insumos).

### Hallazgos con `Base_de_datos_con_Data_Drift_Simulado.xlsx`

- `capital_prestado`: PSI ≈ 0.21, KS ≈ 0.31 (p≈0) — drift significativo.
- `tipo_laboral`: Chi-cuadrado con p-valor prácticamente 0 — la proporción
  entre categorías cambió fuertemente.
- `tipo_credito`: tras redondear cada código al válido más cercano, la
  distribución de letras cambia por completo frente a la referencia
  (referencia: 73% "A", 26% "D"; base actual: 72% "B", 28% "E") — ningún
  valor cae exactamente en los códigos originales (4, 6, 7, 9, 10), así que
  el 100% de los registros se redondeó, y aun así el drift queda claro en
  el Chi-cuadrado por el corrimiento de proporciones.

**Conclusión:** la base simulada presenta drift real y relevante en varias
variables de entrada; en `tipo_credito` el corrimiento es total (100% de
los códigos fuera del rango original), y aun redondeando para poder
comparar, la proporción entre categorías queda muy distinta a la de
entrenamiento. Esto justifica el monitoreo periódico: un modelo entrenado
sobre la población original perdería fiabilidad silenciosamente si esta
base reemplazara a la de producción sin reentrenar o sin revisar el mapeo
de `tipo_credito`.

### Cómo correr la app

```
cd src
streamlit run model_monitoring.py
```

Sube el archivo `.xlsx` **crudo** de la base a monitorear desde la barra
lateral — la limpieza la aplica la app misma.

---

## 6. Despliegue del modelo (API + Docker)

Se despliega **CatBoost**, el modelo con mejor ROC-AUC (0.706) entre los
comparados en la sección 3.

### `exportar_modelo.py`

Entrena CatBoost sobre `X_train` (los mismos datos preprocesados que se
evaluaron en el notebook) y guarda **preprocesador + modelo juntos** en
`models/model.pkl`. Se guardan juntos porque sin el preprocesador ya
ajustado, el modelo no sabría qué hacer con datos crudos nuevos.

```
cd src
python exportar_modelo.py
```

### `model_deploy.py`

API con FastAPI que carga `model.pkl` y expone:

- `GET /saludo` — verificación rápida de que la API está viva.
- `POST /predict` — predicción por lote. Recibe
  `{"registros": [ {...}, {...} ]}`, valida cada registro contra un
  esquema (`RegistroCliente`, con los mismos 25 campos que usa el
  modelo) antes de predecir, y devuelve `prediccion_pago_atiempo` y
  `probabilidad_pago_atiempo` por registro.

Si el modelo no cargó, el lote viene vacío, o algún registro no cumple el
esquema, la API responde con el código HTTP correspondiente (503, 400 o
422) y un detalle del error, en vez de fallar en silencio.

Probado de punta a punta (entrenar → exportar → cargar en la API →
predecir sobre datos reales, incluyendo lote vacío y registro incompleto)
antes de entregarlo.

```
cd src
python model_deploy.py
# o: uvicorn model_deploy:app --host 0.0.0.0 --port 8000
```

### Imagen Docker

`Dockerfile` instala `requirements.txt`, copia `src/` y `models/`, y corre
`uvicorn` como servidor. `.dockerignore` excluye notebooks, datos crudos y
archivos que no hacen falta dentro de la imagen.

```
docker build -t pago-atiempo-api .
docker run -p 8000:8000 pago-atiempo-api
```

La API queda disponible en `http://localhost:8000/predict`.

> **Nota:** hay que correr `exportar_modelo.py` (y tener `models/model.pkl`
> generado) antes de construir la imagen — el Dockerfile no entrena nada,
> solo empaqueta lo que ya existe.

---

## Resumen de decisiones y limitaciones conocidas

- `puntaje` se excluye en todo el proyecto (supervisado, no supervisado y
  monitoreo) por ser fuga de datos.
- El orden temporal para `TimeSeriesSplit` se reconstruye con
  `anio_prestamo`/`mes_prestamo`/`dia_mes`, no con una fecha cruda (no
  existe en el dataset limpio).
- El monitoreo limpia la base actual con la misma lógica de
  `compresion_eda.ipynb` antes de comparar contra la referencia, para que
  el drift se mida en el mismo espacio de variables que usa el modelo. Las
  reglas que eliminan filas (nulos, salario=0) se aplican igual que en el
  entrenamiento, así que un cambio en la *tasa* de nulos entre bases no se
  ve en la tabla de drift — solo en el resumen de limpieza que muestra la
  app.
- Si `tipo_credito` en una base nueva usa códigos fuera del rango visto en
  entrenamiento, se redondea al código válido más cercano en vez de
  fallar o descartarse — con la base de drift simulada, esto le pasó al
  100% de los registros. El drift sigue siendo visible porque cambia la
  proporción entre letras, pero vale la pena recordar que el redondeo
  reduce cuánto se movió el código real (por ejemplo, un 12 y un 9 pueden
  terminar en la misma letra).
- El modelo desplegado (`model.pkl`) se entrena solo con `X_train`, no con
  todo el dataset etiquetado — mismos datos que se evaluaron en la sección
  3. Si se quiere exprimir un poco más de desempeño en producción,
  `exportar_modelo.py` podría reentrenarse con `X_train` + `X_test`
  combinados, ya que la evaluación honesta del modelo ya quedó hecha.
