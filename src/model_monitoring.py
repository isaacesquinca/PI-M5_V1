# src/model_monitoring.py

# ==========================================================
# Librerías
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from scipy.stats import ks_2samp, chi2_contingency
from scipy.spatial.distance import jensenshannon
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from cargar_datos import cargarDatosLimpios
from ft_engineering import preprocesar_datos

st.set_page_config(page_title="Monitoreo de Data Drift", layout="wide")


# ==========================================================
# 1. Referencia: el dataset ya limpio, el mismo con el que se
#    entrenó el modelo.
# ==========================================================

@st.cache_data
def cargar_referencia():
    return cargarDatosLimpios()


@st.cache_data
def cargar_actual_cruda(archivo):
    return pd.read_excel(archivo)


# ==========================================================
# 2. Limpieza de la base actual (misma lógica que
#    compresion_eda.ipynb), para poder comparar limpio vs limpio.
#    No se reutiliza el notebook directamente porque es un
#    notebook de EDA interactivo, no un módulo importable.
# ==========================================================

CODIGOS_TIPO_CREDITO_VALIDOS = np.array([4, 6, 7, 9, 10])
MAPA_TIPO_CREDITO = {4: "A", 6: "B", 7: "C", 9: "D", 10: "E"}


def redondear_a_codigo_valido(valor):
    if pd.isna(valor):
        return np.nan
    return CODIGOS_TIPO_CREDITO_VALIDOS[
        np.argmin(np.abs(CODIGOS_TIPO_CREDITO_VALIDOS - valor))
    ]


def limpiar_datos_actuales(df):
    """Aplica la misma limpieza de compresion_eda.ipynb a una base nueva.

    Devuelve (df_limpio, resumen) donde `resumen` cuenta cuántas filas/valores
    tocó cada regla, para no perder de vista qué tan distinta es la base nueva
    aunque el resultado final ya esté "limpio"."""

    df = df.copy()
    resumen = {"filas_originales": len(df)}

    if "huella_consulta" in df.columns:
        df = df.drop(columns=["huella_consulta"])

    # Outliers conocidos (mismos valores que en compresion_eda.ipynb)
    resumen["outlier_tipo_credito_68"] = int((df["tipo_credito"] == 68).sum())
    df["tipo_credito"] = df["tipo_credito"].replace(68, 6)

    resumen["outlier_edad_121_123"] = int(df["edad_cliente"].isin([121, 122, 123]).sum())
    df["edad_cliente"] = df["edad_cliente"].replace({121: 21, 122: 22, 123: 23})

    resumen["outlier_plazo_90"] = int((df["plazo_meses"] == 90).sum())
    df["plazo_meses"] = df["plazo_meses"].replace(90, 48)

    # Filas con salario_cliente = 0
    resumen["filas_salario_0"] = int((df["salario_cliente"] == 0).sum())
    df = df[df["salario_cliente"] != 0].reset_index(drop=True)

    # Nulos que no se toleran: se elimina la fila
    columnas_sin_nulos = ["puntaje_datacredito", "saldo_mora", "saldo_total"]
    resumen["filas_nulos_eliminadas"] = int(df[columnas_sin_nulos].isna().any(axis=1).sum())
    df = df.dropna(subset=columnas_sin_nulos).reset_index(drop=True)

    # Imputación por mediana (de la propia base actual, igual que en el EDA)
    for col in ["saldo_principal", "saldo_mora_codeudor", "promedio_ingresos_datacredito"]:
        df[col] = df[col].fillna(df[col].median())

    # tendencia_ingresos: valores numéricos -> nulo, luego se imputa con la
    # categoría cuyo promedio de promedio_ingresos_datacredito quede más cerca
    mask_numericos = pd.to_numeric(df["tendencia_ingresos"], errors="coerce").notna()
    resumen["tendencia_ingresos_valores_numericos"] = int(mask_numericos.sum())
    df.loc[mask_numericos, "tendencia_ingresos"] = pd.NA

    promedios = df.groupby("tendencia_ingresos")["promedio_ingresos_datacredito"].mean()

    def imputar_tendencia(ingreso):
        if pd.isna(ingreso):
            return pd.NA
        return (promedios - ingreso).abs().idxmin()

    mask_nulos = df["tendencia_ingresos"].isna()
    df.loc[mask_nulos, "tendencia_ingresos"] = (
        df.loc[mask_nulos, "promedio_ingresos_datacredito"].apply(imputar_tendencia)
    )

    # tipo_credito: se redondea cada valor al código válido más cercano
    # (4, 6, 7, 9, 10 — los mismos de la base original) y luego se mapea a
    # letra. Esto permite comparar limpio vs. limpio aunque los códigos de
    # la base nueva se hayan corrido; se guarda cuántos valores tuvieron que
    # corregirse para no perder de vista qué tanto se movieron.
    valores_antes_redondeo = df["tipo_credito"].copy()
    df["tipo_credito"] = df["tipo_credito"].apply(redondear_a_codigo_valido)
    resumen["tipo_credito_valores_redondeados"] = int(
        (valores_antes_redondeo != df["tipo_credito"]).sum()
    )
    df["tipo_credito"] = df["tipo_credito"].map(MAPA_TIPO_CREDITO)
    df["tipo_credito"] = df["tipo_credito"].fillna("Desconocido")

    # Ingeniería de la fecha
    if "fecha_prestamo" in df.columns:
        df["fecha_prestamo"] = pd.to_datetime(df["fecha_prestamo"])
        df["anio_prestamo"] = df["fecha_prestamo"].dt.year
        df["mes_prestamo"] = df["fecha_prestamo"].dt.month
        df["trimestre"] = df["fecha_prestamo"].dt.quarter
        df["dia_mes"] = df["fecha_prestamo"].dt.day
        df["dia_semana"] = df["fecha_prestamo"].dt.dayofweek
        df["fin_de_semana"] = df["dia_semana"].isin([5, 6])
        df = df.drop(columns=["fecha_prestamo"])

    if "puntaje" in df.columns:
        df = df.drop(columns=["puntaje"])

    resumen["filas_finales"] = len(df)

    return df, resumen


# ==========================================================
# 3. Métricas de data drift
# ==========================================================

def calcular_psi(referencia, actual, bins=10):
    referencia = referencia.dropna()
    actual = actual.dropna()

    cortes = np.histogram_bin_edges(referencia, bins=bins)
    cortes[0], cortes[-1] = -np.inf, np.inf

    ref_pct = np.histogram(referencia, bins=cortes)[0] / len(referencia)
    act_pct = np.histogram(actual, bins=cortes)[0] / len(actual)

    ref_pct = np.clip(ref_pct, 1e-6, None)
    act_pct = np.clip(act_pct, 1e-6, None)

    return float(np.sum((act_pct - ref_pct) * np.log(act_pct / ref_pct)))


def calcular_ks(referencia, actual):
    referencia = referencia.dropna()
    actual = actual.dropna()
    stat, p_valor = ks_2samp(referencia, actual)
    return float(stat), float(p_valor)


def calcular_js(referencia, actual, bins=10):
    referencia = referencia.dropna()
    actual = actual.dropna()

    cortes = np.histogram_bin_edges(pd.concat([referencia, actual]), bins=bins)
    ref_hist = np.histogram(referencia, bins=cortes)[0] / len(referencia)
    act_hist = np.histogram(actual, bins=cortes)[0] / len(actual)

    return float(jensenshannon(ref_hist, act_hist))


def calcular_chi2(referencia, actual):
    referencia = referencia.dropna()
    actual = actual.dropna()

    categorias = sorted(set(referencia.unique()) | set(actual.unique()), key=str)
    tabla = pd.DataFrame({
        "referencia": referencia.value_counts().reindex(categorias, fill_value=0),
        "actual": actual.value_counts().reindex(categorias, fill_value=0),
    })

    stat, p_valor, _, _ = chi2_contingency(tabla.T)
    return float(stat), float(p_valor)


def tabla_resumen_drift(df_ref, df_actual, columnas_numericas, columnas_categoricas):
    filas = []

    for col in columnas_numericas:
        ks_stat, ks_p = calcular_ks(df_ref[col], df_actual[col])
        psi = calcular_psi(df_ref[col], df_actual[col])
        js = calcular_js(df_ref[col], df_actual[col])
        filas.append({
            "variable": col,
            "tipo": "numérica",
            "KS_estadistico": round(ks_stat, 4),
            "KS_p_valor": round(ks_p, 4),
            "PSI": round(psi, 4),
            "JS_divergencia": round(js, 4),
            "drift_detectado": (psi > 0.2) or (ks_p < 0.05),
        })

    for col in columnas_categoricas:
        chi2_stat, chi2_p = calcular_chi2(df_ref[col], df_actual[col])
        filas.append({
            "variable": col,
            "tipo": "categórica",
            "Chi2_estadistico": round(chi2_stat, 4),
            "Chi2_p_valor": round(chi2_p, 4),
            "drift_detectado": chi2_p < 0.05,
        })

    return pd.DataFrame(filas)


# ==========================================================
# 4. Modelo (para la tabla de predicciones)
# ==========================================================

@st.cache_resource
def entrenar_modelo():
    datos = preprocesar_datos()
    modelo = RandomForestClassifier(random_state=42)
    modelo.fit(datos["X_train"], datos["y_train"])
    return modelo, datos["preprocessor"]


# ==========================================================
# 5. App
# ==========================================================

st.title("Monitoreo del modelo y detección de Data Drift")
st.markdown(
    "Compara la base con la que se entrenó el modelo (**referencia**, ya "
    "limpia) contra una base nueva (**actual**) para detectar cambios en la "
    "población que puedan afectar el desempeño del modelo de *Pago a "
    "tiempo*. La base actual se pasa por la misma limpieza que "
    "`compresion_eda.ipynb` antes de compararla, para que la comparación "
    "sea limpio vs. limpio."
)

df_ref = cargar_referencia()

st.sidebar.header("Base de datos actual")
archivo_subido = st.sidebar.file_uploader(
    "Sube la base de datos cruda a monitorear (.xlsx)", type=["xlsx"]
)

if archivo_subido is None:
    st.info("Sube en la barra lateral el archivo .xlsx crudo con los datos a monitorear.")
    st.stop()

df_actual_cruda_completa = cargar_actual_cruda(archivo_subido)

# ----------------------------------------------------------
# Muestreo periódico: se filtra la base CRUDA por fecha antes de
# limpiarla, simulando una corrida de monitoreo con periodicidad
# definida (la fecha cruda ya no existe después de la limpieza).
# ----------------------------------------------------------

if "fecha_prestamo" in df_actual_cruda_completa.columns:
    df_actual_cruda_completa["fecha_prestamo"] = pd.to_datetime(
        df_actual_cruda_completa["fecha_prestamo"]
    )
    fecha_min = df_actual_cruda_completa["fecha_prestamo"].min().date()
    fecha_max = df_actual_cruda_completa["fecha_prestamo"].max().date()

    st.sidebar.header("Muestreo periódico")
    rango_fechas = st.sidebar.date_input(
        "Periodo a analizar",
        value=(fecha_min, fecha_max),
        min_value=fecha_min,
        max_value=fecha_max,
    )

    if len(rango_fechas) == 2:
        inicio, fin = rango_fechas
        mascara = (
            (df_actual_cruda_completa["fecha_prestamo"].dt.date >= inicio)
            & (df_actual_cruda_completa["fecha_prestamo"].dt.date <= fin)
        )
        df_actual_cruda = df_actual_cruda_completa[mascara].reset_index(drop=True)
    else:
        df_actual_cruda = df_actual_cruda_completa
else:
    df_actual_cruda = df_actual_cruda_completa

st.write(
    f"Registros crudos en el periodo seleccionado: **{len(df_actual_cruda)}** "
    f"de {len(df_actual_cruda_completa)} totales"
)

# ----------------------------------------------------------
# Limpieza de la base actual
# ----------------------------------------------------------

df_actual, resumen_limpieza = limpiar_datos_actuales(df_actual_cruda)

with st.expander("Resumen de la limpieza aplicada a la base actual"):
    st.write(resumen_limpieza)
    if resumen_limpieza["tipo_credito_valores_redondeados"] > 0:
        pct = resumen_limpieza["tipo_credito_valores_redondeados"] / resumen_limpieza["filas_finales"] * 100
        st.warning(
            f"{resumen_limpieza['tipo_credito_valores_redondeados']} registros "
            f"({pct:.1f}%) tenían un código de tipo_credito fuera de los "
            "válidos (4, 6, 7, 9, 10) y se redondearon al más cercano antes "
            "de mapear a letra. Un porcentaje alto es señal de drift en esa "
            "variable, aunque ya no se refleje como categoría separada en la "
            "comparación."
        )

# ----------------------------------------------------------
# Resumen de drift (limpio vs. limpio)
# ----------------------------------------------------------

columnas_comunes = [c for c in df_ref.columns if c in df_actual.columns]
df_ref_common = df_ref[columnas_comunes]
df_actual_common = df_actual[columnas_comunes]

columnas_numericas = [
    c for c in df_ref_common.select_dtypes(include=["int64", "float64", "bool"]).columns
    if c != "Pago_atiempo"
]
columnas_categoricas = df_ref_common.select_dtypes(include=["object", "string", "category"]).columns.tolist()

st.subheader("Resumen de Data Drift por variable")
st.caption(
    "PSI > 0.2 o KS con p-valor < 0.05 marcan drift en variables numéricas; "
    "Chi-cuadrado con p-valor < 0.05 lo marca en categóricas."
)

tabla_drift = tabla_resumen_drift(df_ref_common, df_actual_common, columnas_numericas, columnas_categoricas)
st.dataframe(tabla_drift, use_container_width=True)

variables_con_drift = tabla_drift.loc[tabla_drift["drift_detectado"], "variable"].tolist()
if variables_con_drift:
    st.warning(f"Variables con drift detectado: {', '.join(variables_con_drift)}")
else:
    st.success("No se detectó drift significativo en las variables analizadas.")

# ----------------------------------------------------------
# Comparación de distribuciones
# ----------------------------------------------------------

st.subheader("Comparación de distribuciones")

col1, col2 = st.columns(2)

with col1:
    var_numerica = st.selectbox("Variable numérica", columnas_numericas)
    fig, ax = plt.subplots()
    ax.hist(df_ref_common[var_numerica].dropna(), bins=30, alpha=0.5, density=True, label="Referencia")
    ax.hist(df_actual_common[var_numerica].dropna(), bins=30, alpha=0.5, density=True, label="Actual")
    ax.set_title(f"Distribución de {var_numerica}")
    ax.legend()
    st.pyplot(fig)

with col2:
    if columnas_categoricas:
        var_categorica = st.selectbox("Variable categórica", columnas_categoricas)
        prop_ref = df_ref_common[var_categorica].value_counts(normalize=True)
        prop_actual = df_actual_common[var_categorica].value_counts(normalize=True)
        comparacion = pd.DataFrame({"Referencia": prop_ref, "Actual": prop_actual}).fillna(0)
        st.bar_chart(comparacion)
    else:
        st.write("No hay variables categóricas comunes para comparar.")

# ----------------------------------------------------------
# Predicciones del modelo sobre la base actual (ya limpia)
# ----------------------------------------------------------

st.subheader("Predicciones del modelo sobre los datos actuales")

modelo, preprocessor = entrenar_modelo()

df_para_predecir = df_actual.drop(columns=["Pago_atiempo"]) if "Pago_atiempo" in df_actual.columns else df_actual
X_actual = preprocessor.transform(df_para_predecir)

predicciones = modelo.predict(X_actual)
probabilidades = modelo.predict_proba(X_actual)[:, 1]

tabla_predicciones = df_actual.reset_index(drop=True).copy()
tabla_predicciones["prediccion_pago_atiempo"] = predicciones
tabla_predicciones["probabilidad_pago_atiempo"] = probabilidades

st.dataframe(tabla_predicciones.head(200), use_container_width=True)

# ----------------------------------------------------------
# Desempeño real del modelo sobre la base actual (si trae el target)
# ----------------------------------------------------------

if "Pago_atiempo" in df_actual.columns:
    st.subheader("Desempeño del modelo en la base actual")

    y_real = df_actual["Pago_atiempo"].reset_index(drop=True)

    metricas_actuales = {
        "Accuracy": accuracy_score(y_real, predicciones),
        "Precision": precision_score(y_real, predicciones),
        "Recall": recall_score(y_real, predicciones),
        "F1 Score": f1_score(y_real, predicciones),
    }

    st.dataframe(pd.DataFrame([metricas_actuales]).round(4), use_container_width=True)
