# ==========================================================
# ft_engineering.py
# Preprocesamiento de datos
# ==========================================================
 
import pandas as pd
 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
 
from cargar_datos import cargarDatosLimpios
 
 
def preprocesar_datos():
 
    # ======================================================
    # 1. Cargar datos
    # ======================================================
 
    df = cargarDatosLimpios()
 
    print("\n" + "=" * 70)
    print("DATASET LIMPIO")
    print("=" * 70)
    print(df.head())
 
    # ======================================================
    # 2. Separar variables predictoras y objetivo
    #    No existe una columna de fecha cruda: se reconstruye
    #    un orden temporal a partir de anio_prestamo, mes_prestamo
    #    y dia_mes, solo para poder ordenar antes de TimeSeriesSplit.
    #    Estas columnas SÍ se quedan en X como features normales.
    # ======================================================
 
    X = df.drop(columns=["Pago_atiempo"])
    y = df["Pago_atiempo"]
    orden_temporal = (
        df["anio_prestamo"] * 10000 + df["mes_prestamo"] * 100 + df["dia_mes"]
    )
 
    # ======================================================
    # 3. Variables numéricas y categóricas
    #    "bool" se incluye explícitamente: fin_de_semana es
    #    booleana y no calificaba ni como numérica ni como
    #    categórica, así que se perdía en silencio.
    # ======================================================
 
    numeric_features = X.select_dtypes(
        include=["int64", "float64", "bool"]
    ).columns.tolist()
 
    categorical_features = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
 
    print("\nVariables numéricas:")
    print(numeric_features)
 
    print("\nVariables categóricas:")
    print(categorical_features)
 
    # ======================================================
    # 4. Pipeline numérico
    # ======================================================
 
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
 
    # ======================================================
    # 5. Pipeline categórico
    #    sparse_output=False: CatBoost y el gráfico de
    #    importancia de variables necesitan matriz densa
    # ======================================================
 
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
 
    # ======================================================
    # 6. Column Transformer
    # ======================================================
 
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
 
    # ======================================================
    # 7. Train Test Split (se incluye fechas para poder
    #    reordenar X_train por tiempo más adelante)
    # ======================================================
 
    X_train, X_test, y_train, y_test, orden_train, orden_test = train_test_split(
        X,
        y,
        orden_temporal,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
 
    print("\n" + "=" * 70)
    print("DIVISIÓN DEL DATASET")
    print("=" * 70)
    print(f"X_train: {X_train.shape}")
    print(f"X_test : {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test : {y_test.shape}")
 
    # ======================================================
    # 8. Aplicar preprocesamiento
    # ======================================================
 
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)
 
    # ======================================================
    # 9. Obtener nombres de variables
    # ======================================================
 
    feature_names = preprocessor.get_feature_names_out()
 
    print("\n" + "=" * 70)
    print("PREPROCESAMIENTO")
    print("=" * 70)
    print(f"X_train_preprocessed: {X_train_preprocessed.shape}")
    print(f"X_test_preprocessed : {X_test_preprocessed.shape}")
    print(f"\nNúmero de variables finales: {len(feature_names)}")
 
    # ======================================================
    # 10. Retornar todo
    #     y_train/y_test/orden se reindexan (0..n-1) para que
    #     su posición coincida con las filas de los arrays de
    #     numpy ya preprocesados
    # ======================================================
 
    return {
        "X_train": X_train_preprocessed,
        "X_test": X_test_preprocessed,
        "y_train": y_train.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "orden_train": orden_train.reset_index(drop=True),
        "orden_test": orden_test.reset_index(drop=True),
        "preprocessor": preprocessor,
        "feature_names": feature_names,
    }
 
 
# ==========================================================
# Prueba
# ==========================================================
 
if __name__ == "__main__":
    datos = preprocesar_datos()
    print("\nPrimeras variables generadas:\n")
    print(datos["feature_names"][:20])