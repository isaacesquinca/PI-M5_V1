# src/model_deploy.py

# ==========================================================
# Librerías
# ==========================================================

import pickle
from pathlib import Path
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

RUTA_MODELO = Path(__file__).resolve().parent.parent / "models" / "model.pkl"

# ==========================================================
# 1. Inicialización de la aplicación FastAPI
# ==========================================================

app = FastAPI(
    title="API de Predicción de Pago a Tiempo",
    description="Esta API permite predecir si un cliente pagará a tiempo o no.",
    version="1.0.0",
)

# ==========================================================
# 2. Cargar el modelo y el preprocesador (model.pkl)
#    Se guardan juntos en exportar_modelo.py: sin el preprocesador
#    ajustado, el modelo no puede recibir datos crudos.
# ==========================================================

try:
    with open(RUTA_MODELO, "rb") as f:
        objeto_guardado = pickle.load(f)

    preprocessor = objeto_guardado["preprocessor"]
    modelo = objeto_guardado["modelo"]

    print("Modelo cargado exitosamente")

except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    preprocessor = None
    modelo = None


# ==========================================================
# 3. Esquema de entrada (valida los datos antes de predecir,
#    en vez de confiar en lo que llegue por la API)
# ==========================================================

class RegistroCliente(BaseModel):
    tipo_credito: str
    capital_prestado: float
    plazo_meses: int
    edad_cliente: int
    tipo_laboral: str
    salario_cliente: float
    total_otros_prestamos: float
    cuota_pactada: float
    puntaje_datacredito: float
    cant_creditosvigentes: int
    saldo_mora: float
    saldo_total: float
    saldo_principal: float
    saldo_mora_codeudor: float
    creditos_sectorFinanciero: int
    creditos_sectorCooperativo: int
    creditos_sectorReal: int
    promedio_ingresos_datacredito: float
    tendencia_ingresos: str
    anio_prestamo: int
    mes_prestamo: int
    trimestre: int
    dia_mes: int
    dia_semana: int
    fin_de_semana: bool


class LotePredict(BaseModel):
    registros: List[RegistroCliente]


# ==========================================================
# 4. Endpoints
# ==========================================================

@app.get("/saludo")
def saludo():
    return {"mensaje": "API para predecir si un cliente pagará a tiempo."}


@app.post("/predict")
def predict_batch(lote: LotePredict):
    if modelo is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo no está disponible. Revisa los logs del servidor.",
        )

    if not lote.registros:
        raise HTTPException(
            status_code=400,
            detail="No se recibió ningún registro para predecir.",
        )

    try:
        df = pd.DataFrame([registro.dict() for registro in lote.registros])
        X = preprocessor.transform(df)
        predicciones = modelo.predict(X)
        probabilidades = modelo.predict_proba(X)[:, 1]
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Error al procesar los datos recibidos: {e}",
        )

    resultados = [
        {
            "prediccion_pago_atiempo": int(pred),
            "probabilidad_pago_atiempo": float(prob),
        }
        for pred, prob in zip(predicciones, probabilidades)
    ]

    return {"resultados": resultados}


if __name__ == "__main__":
    import uvicorn

    print("\nAbre esto en tu navegador (no la dirección 0.0.0.0 de abajo):")
    print("   http://localhost:8000/docs\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
