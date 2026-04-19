# Respuestas — Práctica Final: Análisis y Modelado de Datos

> Rellena cada pregunta con tu respuesta. Cuando se pida un valor numérico, incluye también una breve explicación de lo que significa.

---

## Ejercicio 1 — Análisis Estadístico Descriptivo
---
He trabajado el dataset de diamantes usando `price` como variable objetivo. Después de eliminar la columna `Unnamed: 0`, el conjunto queda con 53.943 filas y 10 columnas útiles: 7 numéricas y 3 categóricas. Al revisar la estructura, vi que no había valores nulos, así que no fue necesario imputar ni borrar filas por ese motivo. enlace: https://www.kaggle.com/datasets/nancyalaswad90/diamonds-prices

En la parte descriptiva, lo primero que me llamó la atención fue que `price` y `carat` no siguen una distribución simétrica, sino que están claramente sesgadas a la derecha. Es decir, hay muchas observaciones concentradas en valores bajos o medios y una cola de valores altos. En cambio, `depth` y `table` se comportan de forma más estable y compacta.

También detecté valores extremos en `x`, `y` y `z`, incluidos algunos ceros, que no encajan bien con dimensiones físicas reales de un diamante y apuntan a posibles anomalías de medición o registro. Para revisar esto utilicé el método del IQR, porque me parecía más adecuado que otros métodos al tratar con distribuciones asimétricas.

Por último, el análisis de correlaciones dejó bastante claro que `carat`, `x` e `y` son las variables más relacionadas con `price`. Además, aparece una multicolinealidad muy fuerte entre `carat`, `x`, `y` y `z`, algo importante porque luego complica la interpretación del modelo de regresión.

---

**Pregunta 1.1** — ¿De qué fuente proviene el dataset y cuál es la variable objetivo (target)? ¿Por qué tiene sentido hacer regresión sobre ella?

El dataset utilizado procede de Kaggle y corresponde al conjunto “Diamonds Prices 2022”, con información sobre características físicas, geométricas y de calidad de diamantes. Tras eliminar la columna índice sobrante (`Unnamed: 0`), el dataset final queda con 53.943 filas y 10 columnas útiles, de las cuales 3 son categóricas (`cut`, `color`, `clarity`) y 7 numéricas (`carat`, `depth`, `table`, `price`, `x`, `y`, `z`).

He elegido `price` como variable objetivo porque es una variable continua y, además, tiene sentido intentar explicarla a partir de características observables del diamante. En este caso, variables como el quilataje, las dimensiones o la calidad comercial influyen directamente en el precio, así que el problema encaja de forma natural en un enfoque de regresión.

**Pregunta 1.2** — ¿Qué distribución tienen las principales variables numéricas y has encontrado outliers? Indica en qué variables y qué has decidido hacer con ellos.

Las principales variables numéricas no se comportan igual. En mi análisis, `price` y `carat` muestran una asimetría positiva clara: la mayoría de observaciones se concentran en la parte baja, pero hay una cola de valores altos. `depth` y `table`, en cambio, tienen una dispersión más contenida y una forma más compacta. Además, en `x`, `y` y `z` aparecen algunos valores extremos e incluso ceros, lo que sugiere registros poco realistas en dimensiones físicas.

Para detectar outliers utilicé el método del rango intercuartílico (IQR), porque me parecía la opción más robusta en un dataset con distribuciones no simétricas. Los porcentajes de outliers detectados fueron: `price` 6.56%, `depth` 4.72%, `carat` 3.50%, `table` 1.12%, `x` 0.06%, `y` 0.05% y `z` 0.09%.

En esta fase decidí no eliminar esos valores extremos de forma general, porque quería conservar la distribución original del dataset y analizar primero su impacto. En `price` y `carat`, parte de esos extremos pueden ser perfectamente reales. En cambio, en `x`, `y` y `z` sí hay casos con más pinta de anomalía física, así que los dejé señalados y documentados como limitación a tener en cuenta.

**Pregunta 1.3** — ¿Qué tres variables numéricas tienen mayor correlación (en valor absoluto) con la variable objetivo? Indica los coeficientes.

Las tres variables numéricas con mayor correlación en valor absoluto con `price` son `carat` (r = 0.9216), `x` (r = 0.8844) e `y` (r = 0.8654). Esto encaja bastante con lo esperado: el precio del diamante está muy relacionado con su tamaño y sus dimensiones.

Además, al revisar la matriz de correlaciones vi una multicolinealidad muy alta entre varias predictoras: `carat-x` (0.9751), `carat-y` (0.9517), `carat-z` (0.9534), `x-y` (0.9747), `x-z` (0.9708) e `y-z` (0.9520). En la práctica, esto significa que varias de estas variables están aportando información muy parecida, así que luego habrá que ser prudente al interpretar coeficientes del modelo.

**Pregunta 1.4** — ¿Hay valores nulos en el dataset?.

No encontré valores nulos en el dataset. El porcentaje de valores faltantes es 0% en todas las columnas (`carat`, `cut`, `color`, `clarity`, `depth`, `table`, `price`, `x`, `y`, `z`), así que no fue necesario aplicar ningún tratamiento de imputación ni eliminar registros por este motivo.

---

## Ejercicio 2 — Inferencia con Scikit-Learn

---
Para este ejercicio usé el mismo dataset del apartado anterior y mantuve `price` como variable objetivo. Separé los predictores en numéricos y categóricos y monté un pipeline con imputación, escalado en las variables numéricas y codificación one-hot en las categóricas. Después hice la partición train/test en proporción 80/20 con `random_state=42`.

El modelo de regresión lineal dio un resultado bastante bueno en test: `R² = 0.9222`, `MAE = 731.5479` y `RMSE = 1102.1477`. Viendo los resultados junto con el análisis del Ejercicio 1, tiene sentido que `carat` aparezca como variable muy influyente. También se entiende mejor por qué algunos coeficientes no son tan intuitivos: la multicolinealidad entre `carat`, `x`, `y` y `z` hace que varias variables se repartan información muy parecida.

---

**Pregunta 2.1** — Indica los valores de MAE, RMSE y R² de la regresión lineal sobre el test set. ¿El modelo funciona bien? ¿Por qué?

En el conjunto de test, la regresión lineal ha obtenido estos valores: `MAE = 731.5479`, `RMSE = 1102.1477` y `R² = 0.9222`.

En general, considero que el modelo funciona bien porque consigue explicar el 92.22% de la variabilidad de `price` en los datos de prueba. El error absoluto medio ronda los 731.55 dólares, mientras que el RMSE sube hasta 1102.15 dólares, lo que indica que hay algunos errores más grandes que pesan más en la métrica.

No veo señales claras de underfitting, ya que el ajuste es alto, y tampoco puedo afirmar overfitting solo con esto porque el rendimiento en test sigue siendo fuerte. Las variables que más peso parecen tener son `carat`, varias categorías de `clarity` y `color`, y también `x`. Aun así, no conviene leer esos coeficientes de forma literal sin tener en cuenta la multicolinealidad detectada antes.


---

## Ejercicio 3 — Regresión Lineal Múltiple en NumPy

---
En este ejercicio implementé una regresión lineal múltiple desde cero usando solo NumPy, sin apoyarme en Scikit-Learn para el ajuste. Añadí el intercepto con una columna de unos, resolví los coeficientes con `lstsq`, generé las predicciones y calculé después `MAE`, `RMSE` y `R²` también de forma manual.

Los coeficientes ajustados salen bastante cerca de los valores reales con los que se generaron los datos sintéticos, así que la implementación parece correcta. En mi caso, `MAE` y `RMSE` quedan muy cerca del rango orientativo del enunciado, mientras que `R²` sale algo más bajo de lo esperado. Aun así, el comportamiento general del modelo es coherente y el ejercicio cumple su objetivo, que era entender y programar el ajuste OLS desde dentro.
---

**Pregunta 3.1** — Explica en tus propias palabras qué hace la fórmula β = (XᵀX)⁻¹ Xᵀy y por qué es necesario añadir una columna de unos a la matriz X.

> La fórmula β = (XᵀX)⁻¹ Xᵀy sirve para calcular los coeficientes de la regresión lineal múltiple mediante mínimos cuadrados ordinarios. Lo que hace, en la práctica, es encontrar los valores de β que mejor ajustan la relación entre las variables predictoras y la variable objetivo minimizando el error cuadrático total.

La columna de unos en la matriz `X` es necesaria para que el modelo pueda estimar el intercepto `β₀`. Si no se añade, el modelo queda forzado a pasar por el origen y pierde el término independiente, lo que en muchos casos da un ajuste menos realista.

**Pregunta 3.2** — Copia aquí los cuatro coeficientes ajustados por tu función y compáralos con los valores de referencia del enunciado.

| Parametro | Valor real | Valor ajustado |
|-----------|-----------|----------------|
| β₀        | 5.0       | 4.8650         |
| β₁        | 2.0       | 2.0636         |
| β₂        | -1.0      | -1.1170        |
| β₃        | 0.5       | 0.4385         |

> Los coeficientes ajustados son bastante próximos a los valores reales de referencia. Hay pequeñas diferencias, pero el ajuste reproduce bien la estructura del modelo con el que se generaron los datos. Eso me indica que la implementación de OLS con NumPy está funcionando correctamente.

**Pregunta 3.3** — ¿Qué valores de MAE, RMSE y R² has obtenido? ¿Se aproximan a los de referencia?

Los valores obtenidos han sido: `MAE = 1.1665`, `RMSE = 1.4612` y `R² = 0.6897`.

El `MAE` y el `RMSE` encajan bien con los valores orientativos del enunciado, ya que están muy cerca de 1.20 y 1.50. El `R²` me ha quedado algo más bajo que el valor de referencia aproximado, pero los coeficientes siguen siendo cercanos a los reales y el error de predicción es bajo. Por eso considero que la implementación es válida aunque no replique exactamente todos los valores esperados.

**Pregunta 3.4* — Compara los resultados con la reacción logística anterior para tu dataset y comprueba si el resultado es parecido. Explica qué ha sucedido. 

No se pueden comparar de forma directa los resultados numéricos porque el contexto cambia bastante. En el Ejercicio 2 trabajé con un dataset real de diamantes, con variables numéricas y categóricas, además de todo el preprocesamiento asociado. En el Ejercicio 3, en cambio, los datos son sintéticos, solo hay tres variables predictoras y el ajuste está hecho manualmente con NumPy.

Aun así, sí veo una relación conceptual clara entre ambos ejercicios: en los dos se intenta predecir una variable continua y se evalúa el rendimiento con `MAE`, `RMSE` y `R²`. La diferencia importante no está tanto en la idea del modelo como en la forma de implementarlo.

---

## Ejercicio 4 — Series Temporales

---
En este ejercicio analicé la serie temporal sintética diaria entre 2018-01-01 y 2023-12-31 y su descomposición aditiva con periodo 365. Al visualizarla, se aprecia una tendencia creciente bastante clara, una estacionalidad anual marcada y también una oscilación más lenta de largo plazo.

Al revisar el residuo, la media queda muy cerca de cero y la forma general encaja bastante bien con una distribución aproximadamente normal. Además, el test de Jarque-Bera no rechaza normalidad y el test ADF confirma estacionariedad. Con todo eso, el residuo se puede considerar razonablemente compatible con un “ruido ideal” en términos prácticos.

---

**Pregunta 4.1** — ¿La serie presenta tendencia? Descríbela brevemente (tipo, dirección, magnitud aproximada).

Sí, la serie presenta una tendencia creciente y aproximadamente lineal. A lo largo del periodo completo, el nivel de la serie aumenta alrededor de 110 unidades, lo que equivale aproximadamente a una pendiente media de 0.05 unidades por día. En resumen, la serie va subiendo de forma sostenida con el paso del tiempo.

**Pregunta 4.2** — ¿Hay estacionalidad? Indica el periodo aproximado en días y la amplitud del patrón estacional.

Sí, hay una estacionalidad clara con un periodo aproximado de 365 días. El patrón se repite de forma anual y la amplitud es de unas 32 unidades pico a pico, es decir, alrededor de ±16 unidades respecto al nivel medio estacional.

**Pregunta 4.3** — ¿Se aprecian ciclos de largo plazo en la serie? ¿Cómo los diferencias de la tendencia?

Sí, además de la tendencia se aprecia un ciclo de largo plazo con un periodo cercano a 1461 días, es decir, unos 4 años. Lo distingo de la tendencia porque la tendencia mantiene una subida global y bastante regular, mientras que el ciclo introduce oscilaciones lentas de subida y bajada alrededor de esa trayectoria general.

**Pregunta 4.4** — ¿El residuo se ajusta a un ruido ideal? Indica la media, la desviación típica y el resultado del test de normalidad (p-value) para justificar tu respuesta.

El residuo se ajusta bastante bien a lo que cabría esperar de un ruido ideal. La media es `0.127078` y la desviación típica `3.222043`, así que está centrado cerca de cero y con una dispersión moderada. Además, la asimetría (`-0.050917`) y la curtosis (`-0.061028`) son muy próximas a cero, lo que refuerza la idea de una forma aproximadamente normal.

En el test de normalidad Jarque-Bera sale un `p-value = 0.576561`, por lo que no se rechaza la hipótesis de normalidad. Por su parte, el test ADF da un `p-value = 0.000000`, lo que indica que el residuo es estacionario. Con estos resultados, me parece razonable interpretar el residuo como un ruido aproximadamente gaussiano y sin tendencia.

---

*Fin del documento de respuestas*
