# Predicción de Pago a Tiempo — Carga de Datos y EDA

Este README cubre únicamente la primera etapa del proyecto:
`cargar_datos.py` y `compresion_eda.ipynb` (carga, limpieza y análisis
exploratorio). No incluye modelado ni monitoreo.

## Caso de negocio

Una entidad de crédito necesita anticipar si un cliente pagará a tiempo un
préstamo (`Pago_atiempo`). Antes de modelar, hay que asegurar que los datos
estén limpios y entender qué variables realmente se relacionan con el
resultado — y detectar a tiempo cualquier variable que en realidad sea fuga
de información en vez de un predictor legítimo.

## Archivos

- **`cargar_datos.py`**
  - `cargarDatos()`: lee `Base_de_datos.xlsx` (datos crudos).

## Proceso de limpieza

- Se eliminó `huella_consulta` (identificador sin valor predictivo).
- Se corrigieron outliers de captura: `tipo_credito=68`, `edad_cliente` en
  121-123, `plazo_meses=90`.
- Se eliminaron filas con `salario_cliente=0` y filas con nulos en
  `puntaje_datacredito`, `saldo_mora` o `saldo_total`.
- Se imputó con mediana `saldo_principal`, `saldo_mora_codeudor` y
  `promedio_ingresos_datacredito`.
- `tendencia_ingresos` se limpió reasignando valores numéricos inválidos a
  la categoría más cercana según `promedio_ingresos_datacredito`.
- `tipo_credito` se recodificó de número a categoría (A-E).
- `fecha_prestamo` se descompuso en `anio_prestamo`, `mes_prestamo`,
  `trimestre`, `dia_mes`, `dia_semana` y `fin_de_semana`, y se eliminó la
  fecha cruda.

## Principales hallazgos del EDA

Comparando clientes que pagan a tiempo vs. los que no:

| Variable | No pagan | Pagan | Diferencia |
|---|---|---|---|
| `puntaje` (interno) | 23.38 | 94.57 | +304.5% |
| `salario_cliente` | $4.18M | $7.92M | +89.2% |
| `saldo_mora` | 82.53 | 4.00 | −95.2% |
| `puntaje_datacredito` | 763.62 | 792.89 | +3.8% |
| `capital_prestado` | $2.76M | $2.36M | −14.3% |
| `plazo_meses` | 12.55 | 10.50 | −16.3% |
| `promedio_ingresos_datacredito` | $1.47M | $1.81M | +23.3% |
| `edad_cliente` | 40.0 años | 43.0 años | +7.4% |

Variables con mayor capacidad predictiva aparente: `puntaje`,
`salario_cliente`, `saldo_mora`, `promedio_ingresos_datacredito`,
`capital_prestado`, `plazo_meses`, `puntaje_datacredito`, `saldo_total`.

### Hallazgo clave: fuga de datos en `puntaje`

La matriz de correlación mostró que `puntaje` tiene 0.92 de correlación con
`Pago_atiempo` — muy por encima del resto. Por esa magnitud, se decidió
tratarla como fuga de información (probablemente se genera después de
conocerse el comportamiento de pago, no al momento de originar el crédito)
y **se eliminó del dataset**.

### Multicolinealidad detectada

Tres pares de variables muestran relación fuerte entre sí y deben tenerse
en cuenta en el modelado (redundancia, sensibilidad en modelos lineales):
- `capital_prestado` – `cuota_pactada`
- `cant_creditosvigentes` – `creditos_sectorFinanciero`
- `saldo_total` – `saldo_principal`

## Cómo correr

```bash
pip install -r requirements.txt
python cargar_datos.py          # prueba rápida de carga
jupyter notebook compresion_eda.ipynb   # limpieza + EDA completo
```