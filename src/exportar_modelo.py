# src/exportar_modelo.py
#
# Entrena el mejor modelo (CatBoost, el de mayor ROC-AUC en
# model_training_evaluation.ipynb) y lo exporta junto al preprocesador
# para que model_deploy.py pueda cargarlos y predecir.

import os
import pickle

from catboost import CatBoostClassifier

from ft_engineering import preprocesar_datos


def exportar_modelo():
    datos = preprocesar_datos()

    modelo = CatBoostClassifier(random_state=42, verbose=0)
    modelo.fit(datos["X_train"], datos["y_train"])

    ruta_modelos = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "models"
    )
    os.makedirs(ruta_modelos, exist_ok=True)
    ruta_pkl = os.path.join(ruta_modelos, "model.pkl")

    with open(ruta_pkl, "wb") as f:
        pickle.dump({"preprocessor": datos["preprocessor"], "modelo": modelo}, f)

    print(f"Modelo exportado en: {ruta_pkl}")


if __name__ == "__main__":
    exportar_modelo()
