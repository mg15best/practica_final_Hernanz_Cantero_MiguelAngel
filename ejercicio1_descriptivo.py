import os
import pandas as pd

# Crear carpeta output si no existe
os.makedirs("output", exist_ok=True)

# Cargar dataset
df = pd.read_csv("data/diamonds_prices_2022.csv")

# Eliminar columna sobrante si existe
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Variable objetivo
target = "price"

# Columnas numéricas y categóricas
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()

# ==================================================
# RESUMEN ESTRUCTURAL
# ==================================================
print("=" * 60)
print("RESUMEN ESTRUCTURAL DEL DATASET")
print("=" * 60)
print(f"Número de filas: {df.shape[0]}")
print(f"Número de columnas: {df.shape[1]}")
print(f"Tamaño en memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\nTipos de dato:")
print(df.dtypes)

print("\nPorcentaje de valores nulos por columna:")
null_percent = (df.isnull().sum() / len(df)) * 100
print(null_percent)

print("\nVariables numéricas:")
print(numeric_cols)

print("\nVariables categóricas:")
print(categorical_cols)

print("\nVariable objetivo:")
print(target)

# ==================================================
# ESTADÍSTICOS DESCRIPTIVOS COMPLETOS
# ==================================================
estadisticos = pd.DataFrame(index=numeric_cols)

estadisticos["count"] = df[numeric_cols].count()
estadisticos["mean"] = df[numeric_cols].mean()
estadisticos["median"] = df[numeric_cols].median()
estadisticos["mode"] = df[numeric_cols].mode().iloc[0]
estadisticos["std"] = df[numeric_cols].std()
estadisticos["var"] = df[numeric_cols].var()
estadisticos["min"] = df[numeric_cols].min()
estadisticos["q1"] = df[numeric_cols].quantile(0.25)
estadisticos["q2"] = df[numeric_cols].quantile(0.50)
estadisticos["q3"] = df[numeric_cols].quantile(0.75)
estadisticos["max"] = df[numeric_cols].max()

# Guardar CSV
estadisticos.to_csv("output/ej1_descriptivo.csv", encoding="utf-8")

print("\n" + "=" * 60)
print("ESTADÍSTICOS DESCRIPTIVOS COMPLETOS")
print("=" * 60)
print(estadisticos.round(4))

print("\nArchivo guardado correctamente:")
print("output/ej1_descriptivo.csv")

# ==================================================
# ESTADÍSTICOS ESPECÍFICOS DE LA VARIABLE OBJETIVO
# ==================================================
q1_target = df[target].quantile(0.25)
q3_target = df[target].quantile(0.75)
iqr_target = q3_target - q1_target
skew_target = df[target].skew()
kurt_target = df[target].kurtosis()

print("\n" + "=" * 60)
print("ESTADÍSTICOS DE LA VARIABLE OBJETIVO")
print("=" * 60)
print(f"Variable objetivo: {target}")
print(f"Q1: {q1_target:.2f}")
print(f"Q3: {q3_target:.2f}")
print(f"IQR: {iqr_target:.2f}")
print(f"Asimetría (skewness): {skew_target:.4f}")
print(f"Curtosis: {kurt_target:.4f}")

import math
import matplotlib.pyplot as plt
import seaborn as sns

# ==================================================
# HISTOGRAMAS DE VARIABLES NUMÉRICAS
# ==================================================
n_cols = 2
n_rows = math.ceil(len(numeric_cols) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    sns.histplot(df[col], kde=True, ax=axes[i])
    axes[i].set_title(f"Histograma de {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Frecuencia")

# Ocultar ejes sobrantes si los hay
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig("output/ej1_histogramas.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nArchivo guardado correctamente:")
print("output/ej1_histogramas.png")

# ==================================================
# BOXPLOTS DE LA VARIABLE OBJETIVO POR CATEGÓRICAS
# ==================================================
fig, axes = plt.subplots(1, len(categorical_cols), figsize=(18, 5))

for i, col in enumerate(categorical_cols):
    sns.boxplot(data=df, x=col, y=target, ax=axes[i])
    axes[i].set_title(f"{target} por {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel(target)
    axes[i].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("output/ej1_boxplots.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nArchivo guardado correctamente:")
print("output/ej1_boxplots.png")

# ==================================================
# DETECCIÓN DE OUTLIERS CON IQR
# ==================================================
outliers_info = []

for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    mask_outliers = (df[col] < lower) | (df[col] > upper)
    n_outliers = mask_outliers.sum()
    pct_outliers = (n_outliers / len(df)) * 100

    outliers_info.append({
        "variable": col,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower,
        "upper_bound": upper,
        "n_outliers": n_outliers,
        "pct_outliers": pct_outliers
    })

outliers_df = pd.DataFrame(outliers_info)

print("\n" + "=" * 60)
print("DETECCIÓN DE OUTLIERS (MÉTODO IQR)")
print("=" * 60)
print(outliers_df.round(4))

with open("output/ej1_outliers.txt", "w", encoding="utf-8") as f:
    f.write("DETECCIÓN DE OUTLIERS (MÉTODO IQR)\n")
    f.write("=" * 60 + "\n")
    f.write(outliers_df.round(4).to_string(index=False))

print("\nArchivo guardado correctamente:")
print("output/ej1_outliers.txt")

# ==================================================
# VARIABLES CATEGÓRICAS: FRECUENCIAS Y GRÁFICOS
# ==================================================
fig, axes = plt.subplots(1, len(categorical_cols), figsize=(18, 5))

for i, col in enumerate(categorical_cols):
    freq_abs = df[col].value_counts()
    freq_rel = df[col].value_counts(normalize=True) * 100

    print("\n" + "=" * 60)
    print(f"FRECUENCIAS DE {col.upper()}")
    print("=" * 60)
    print("Frecuencia absoluta:")
    print(freq_abs)
    print("\nFrecuencia relativa (%):")
    print(freq_rel.round(2))

    sns.countplot(data=df, x=col, order=freq_abs.index, ax=axes[i])
    axes[i].set_title(f"Frecuencia de {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Conteo")
    axes[i].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("output/ej1_categoricas.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nArchivo guardado correctamente:")
print("output/ej1_categoricas.png")

# ==================================================
# CORRELACIONES Y HEATMAP
# ==================================================
corr_matrix = df[numeric_cols].corr(method="pearson")

print("\n" + "=" * 60)
print("MATRIZ DE CORRELACIONES DE PEARSON")
print("=" * 60)
print(corr_matrix.round(4))

# Guardar heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Mapa de calor de correlaciones")
plt.tight_layout()
plt.savefig("output/ej1_heatmap_correlacion.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nArchivo guardado correctamente:")
print("output/ej1_heatmap_correlacion.png")

# Top 3 correlaciones absolutas con la variable objetivo
target_corr = corr_matrix[target].drop(target).abs().sort_values(ascending=False)

print("\n" + "=" * 60)
print(f"TOP 3 CORRELACIONES ABSOLUTAS CON {target.upper()}")
print("=" * 60)
print(target_corr.head(3).round(4))

# Detección de multicolinealidad entre predictoras
multicol_pairs = []

predictor_corr = corr_matrix.drop(index=target, columns=target)

cols = predictor_corr.columns
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        r = predictor_corr.iloc[i, j]
        if abs(r) > 0.9:
            multicol_pairs.append((cols[i], cols[j], r))

print("\n" + "=" * 60)
print("PARES CON POSIBLE MULTICOLINEALIDAD (|r| > 0.9)")
print("=" * 60)

if multicol_pairs:
    for var1, var2, r in multicol_pairs:
        print(f"{var1} - {var2}: r = {r:.4f}")
else:
    print("No se detectaron pares con |r| > 0.9")