import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==================================================
# CONFIGURACIÓN INICIAL
# ==================================================
os.makedirs("output", exist_ok=True)

DATA_PATH = "data/diamonds_prices_2022.csv"
TARGET = "price"


def cargar_datos(ruta):
    """
    Carga el dataset y elimina la columna sobrante 'Unnamed: 0' si existe.

    Parámetros
    ----------
    ruta : str
        Ruta al archivo CSV.

    Retorna
    -------
    pd.DataFrame
        DataFrame limpio de la columna índice sobrante.
    """
    df = pd.read_csv(ruta)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    return df


def preparar_datos(df, target):
    """
    Separa el dataset en variables predictoras (X) y variable objetivo (y),
    identificando también columnas numéricas y categóricas.

    Parámetros
    ----------
    df : pd.DataFrame
        Dataset completo.
    target : str
        Nombre de la variable objetivo.

    Retorna
    -------
    tuple
        X, y, numeric_cols, categorical_cols
    """
    X = df.drop(columns=[target])
    y = df[target]

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "str"]).columns.tolist()

    return X, y, numeric_cols, categorical_cols


def construir_pipeline(numeric_cols, categorical_cols):
    """
    Construye un pipeline de preprocesamiento + regresión lineal.

    - Numéricas: imputación por mediana + escalado
    - Categóricas: imputación por moda + one-hot encoding

    Parámetros
    ----------
    numeric_cols : list
        Lista de columnas numéricas.
    categorical_cols : list
        Lista de columnas categóricas.

    Retorna
    -------
    Pipeline
        Pipeline completo listo para entrenar.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    return model


def guardar_metricas(mae, rmse, r2, ruta_salida):
    """
    Guarda las métricas del modelo en un archivo de texto.

    Parámetros
    ----------
    mae : float
        Mean Absolute Error.
    rmse : float
        Root Mean Squared Error.
    r2 : float
        Coeficiente de determinación R².
    ruta_salida : str
        Ruta del archivo txt.
    """
    with open(ruta_salida, "w", encoding="utf-8") as f:
        f.write("Ejercicio 2 — Métricas de Regresión Lineal\n")
        f.write("=" * 50 + "\n")
        f.write(f"MAE  : {mae:.4f}\n")
        f.write(f"RMSE : {rmse:.4f}\n")
        f.write(f"R²   : {r2:.4f}\n")


def graficar_residuos(y_test, y_pred, ruta_salida):
    """
    Genera y guarda el gráfico de residuos.

    Parámetros
    ----------
    y_test : pd.Series o np.ndarray
        Valores reales.
    y_pred : np.ndarray
        Valores predichos.
    ruta_salida : str
        Ruta de salida de la imagen.
    """
    residuos = y_test - y_pred

    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuos, alpha=0.5)
    plt.axhline(y=0, linestyle="--")
    plt.xlabel("Valores predichos")
    plt.ylabel("Residuos")
    plt.title("Gráfico de residuos — Regresión lineal")
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150, bbox_inches="tight")
    plt.close()


def obtener_coeficientes(model, numeric_cols, categorical_cols):
    """
    Extrae los coeficientes del modelo entrenado y los ordena por valor absoluto.

    Parámetros
    ----------
    model : Pipeline
        Pipeline ya entrenado.
    numeric_cols : list
        Columnas numéricas originales.
    categorical_cols : list
        Columnas categóricas originales.

    Retorna
    -------
    pd.DataFrame
        Tabla con variables y coeficientes ordenados por impacto absoluto.
    """
    preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["regressor"]

    feature_names = preprocessor.get_feature_names_out()
    coefs = regressor.coef_

    coef_df = pd.DataFrame({
        "variable": feature_names,
        "coeficiente": coefs
    })

    coef_df["abs_coef"] = coef_df["coeficiente"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False)

    return coef_df


if __name__ == "__main__":
    # 1. Cargar datos
    df = cargar_datos(DATA_PATH)

    # 2. Separar X e y
    X, y, numeric_cols, categorical_cols = preparar_datos(df, TARGET)

    print("=" * 60)
    print("EJERCICIO 2 — INFERENCIA CON SCIKIT-LEARN")
    print("=" * 60)
    print(f"Filas: {df.shape[0]}")
    print(f"Columnas predictoras: {X.shape[1]}")
    print(f"Variable objetivo: {TARGET}")
    print(f"Variables numéricas: {numeric_cols}")
    print(f"Variables categóricas: {categorical_cols}")

    # 3. Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print("\nTamaño train/test:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test : {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test : {y_test.shape}")

    # 4. Construir y entrenar modelo
    model = construir_pipeline(numeric_cols, categorical_cols)
    model.fit(X_train, y_train)

    # 5. Predicción
    y_pred = model.predict(X_test)

    # 6. Métricas
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    print("\nMétricas del modelo:")
    print(f"MAE  = {mae:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"R²   = {r2:.4f}")

    # 7. Guardar métricas y gráfico
    guardar_metricas(mae, rmse, r2, "output/ej2_metricas_regresion.txt")
    graficar_residuos(y_test, y_pred, "output/ej2_residuos.png")

    print("\nArchivos guardados correctamente:")
    print("output/ej2_metricas_regresion.txt")
    print("output/ej2_residuos.png")

    # 8. Mostrar variables más influyentes
    coef_df = obtener_coeficientes(model, numeric_cols, categorical_cols)

    print("\nTop 10 variables más influyentes (por valor absoluto del coeficiente):")
    print(coef_df.head(10)[["variable", "coeficiente"]].round(4))

    # 9. Gráfico de coeficientes más influyentes
    top_coef = coef_df.head(10).copy()
    top_coef = top_coef.sort_values("coeficiente")
    
    plt.figure(figsize=(10, 6))
    plt.barh(top_coef["variable"], top_coef["coeficiente"])
    plt.xlabel("Coeficiente")
    plt.ylabel("Variable")
    plt.title("Top 10 coeficientes más influyentes")
    plt.tight_layout()
    plt.savefig("output/ej2_coeficientes.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    print("output/ej2_coeficientes.png")
