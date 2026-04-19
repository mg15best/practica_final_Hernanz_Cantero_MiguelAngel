# Respuestas — Práctica Final: Análisis y Modelado de Datos

> Rellena cada pregunta con tu respuesta. Cuando se pida un valor numérico, incluye también una breve explicación de lo que significa.

---

## Ejercicio 1 — Análisis Estadístico Descriptivo
---
Se ha realizado un análisis estadístico descriptivo completo sobre el dataset “Diamonds Prices 2022” de Kaggle, utilizando `price` como variable objetivo. Tras eliminar la columna índice sobrante (`Unnamed: 0`), el conjunto final quedó compuesto por 53.943 observaciones y 10 variables útiles, con 7 variables numéricas (`carat`, `depth`, `table`, `price`, `x`, `y`, `z`) y 3 categóricas (`cut`, `color`, `clarity`).

El análisis estructural mostró que no existen valores nulos en ninguna columna, por lo que no fue necesario aplicar imputación ni eliminación de registros por ausencia de datos. En las variables numéricas se observaron distribuciones diferentes: `price` y `carat` presentaron asimetría positiva, mientras que `depth` y `table` mostraron distribuciones más compactas. Además, en `x`, `y` y `z` aparecieron valores extremos y algunos mínimos iguales a 0, lo que sugiere posibles anomalías en dimensiones físicas.

La detección de outliers mediante el método IQR confirmó la presencia de valores atípicos en todas las variables numéricas, especialmente en `price` (6.56%), `depth` (4.72%) y `carat` (3.50%). En esta fase se optó por no eliminarlos, ya que parte de ellos puede corresponder a diamantes reales de alto valor o tamaño, aunque quedaron documentados para su interpretación posterior.

Por último, el análisis de correlaciones mostró que `carat`, `x` e `y` son las variables más relacionadas con `price`, y también reveló una multicolinealidad muy alta entre `carat`, `x`, `y` y `z`, aspecto importante para interpretar después el modelo de regresión del Ejercicio 2.

---

**Pregunta 1.1** — ¿De qué fuente proviene el dataset y cuál es la variable objetivo (target)? ¿Por qué tiene sentido hacer regresión sobre ella?

> > El dataset utilizado procede de Kaggle y corresponde al conjunto “Diamonds Prices 2022”, que contiene información sobre diamantes, incluyendo variables físicas, geométricas y de calidad comercial. Tras eliminar la columna índice sobrante (`Unnamed: 0`), el conjunto final cuenta con 53.943 filas y 10 columnas útiles, con 3 variables categóricas (`cut`, `color`, `clarity`) y 7 variables numéricas (`carat`, `depth`, `table`, `price`, `x`, `y`, `z`), por lo que cumple los requisitos del ejercicio.

La variable objetivo seleccionada es `price`, ya que es una variable cuantitativa continua. Tiene sentido aplicar regresión sobre ella porque el precio de un diamante depende de forma directa de características medibles y observables del propio diamante, como su quilataje, dimensiones, proporciones y nivel de calidad. En consecuencia, el objetivo del modelo será estimar un valor continuo a partir de un conjunto de variables predictoras numéricas y categóricas.

**Pregunta 1.2** — ¿Qué distribución tienen las principales variables numéricas y has encontrado outliers? Indica en qué variables y qué has decidido hacer con ellos.

> Las principales variables numéricas muestran comportamientos distintos. `price` y `carat` presentan una distribución asimétrica positiva, con concentración de valores en la parte baja y una cola hacia valores altos. En el caso de `depth` y `table`, la dispersión es menor y las distribuciones son más compactas. Además, en `x`, `y` y `z` aparecen valores mínimos iguales a 0 o muy extremos, lo que sugiere posibles registros anómalos en dimensiones físicas del diamante.

Para detectar outliers se ha utilizado el método del rango intercuartílico (IQR), ya que es robusto frente a distribuciones asimétricas y resulta apropiado para este dataset. Los porcentajes de outliers detectados han sido: `price` 6.56%, `depth` 4.72%, `carat` 3.50%, `table` 1.12%, `x` 0.06%, `y` 0.05% y `z` 0.09%.

La decisión tomada ha sido no eliminar los outliers en esta fase descriptiva general, sino detectarlos y documentarlos. Esto se debe a que parte de los valores extremos de `price` y `carat` pueden corresponder a diamantes reales de alto valor o tamaño, mientras que en `x`, `y` y `z` sí existen observaciones con mayor apariencia de anomalía física. Por tanto, los outliers se han identificado y justificado, pero se conservan de momento para no alterar artificialmente la distribución original de los datos.

**Pregunta 1.3** — ¿Qué tres variables numéricas tienen mayor correlación (en valor absoluto) con la variable objetivo? Indica los coeficientes.

> Las tres variables numéricas con mayor correlación en valor absoluto con la variable objetivo `price` son `carat` (r = 0.9216), `x` (r = 0.8844) e `y` (r = 0.8654). Esto indica que el precio del diamante está fuertemente relacionado con su peso y con sus dimensiones físicas.

Además, se detecta posible multicolinealidad entre varias predictoras, ya que existen pares con correlaciones superiores a 0.9: `carat-x` (0.9751), `carat-y` (0.9517), `carat-z` (0.9534), `x-y` (0.9747), `x-z` (0.9708) e `y-z` (0.9520). Esto sugiere que varias variables geométricas aportan información muy similar, por lo que más adelante habrá que tenerlo en cuenta al interpretar el modelo de regresión.

**Pregunta 1.4** — ¿Hay valores nulos en el dataset? ¿Qué porcentaje representan y cómo los has tratado?

> No se han encontrado valores nulos en el dataset. El porcentaje de valores faltantes es 0% en todas las columnas (`carat`, `cut`, `color`, `clarity`, `depth`, `table`, `price`, `x`, `y`, `z`). Por tanto, no ha sido necesario aplicar ningún tratamiento de imputación ni eliminación de registros por ausencia de datos.

---

## Ejercicio 2 — Inferencia con Scikit-Learn

---
Para el preprocesamiento se ha utilizado el mismo dataset del Ejercicio 1. La variable objetivo ha sido `price`, mientras que las variables predictoras se han dividido en numéricas (`carat`, `depth`, `table`, `x`, `y`, `z`) y categóricas (`cut`, `color`, `clarity`). Las variables categóricas se han transformado mediante OneHotEncoder, mientras que las variables numéricas se han escalado con StandardScaler. Aunque el dataset no presenta valores nulos, se ha incorporado igualmente una imputación estándar en el pipeline por robustez y reproducibilidad.

Posteriormente, los datos se han dividido en entrenamiento y prueba con una proporción 80/20 y `random_state=42`, tal como exige el enunciado. Sobre esta base se ha entrenado un modelo de regresión lineal con Scikit-Learn.

El modelo ha alcanzado un R² de 0.9222, con un MAE de 731.5479 y un RMSE de 1102.1477, lo que indica un rendimiento predictivo elevado. El precio del diamante se explica en gran medida por su quilataje y por variables relacionadas con la calidad, especialmente `clarity` y `color`. Sin embargo, la interpretación de algunos coeficientes debe realizarse con cautela porque en el análisis descriptivo ya se detectó una multicolinealidad muy alta entre `carat`, `x`, `y` y `z`.

La información más útil del Ejercicio 1 para interpretar el modelo ha sido, en primer lugar, la fuerte correlación de `carat`, `x` e `y` con `price`, ya que anticipaba que estas variables serían muy relevantes en la predicción. En segundo lugar, la detección de multicolinealidad entre `carat`, `x`, `y` y `z` ayuda a entender por qué algunos coeficientes del modelo pueden presentar signos o magnitudes menos intuitivas de lo esperado. Por último, la asimetría positiva y la presencia de outliers en `price` permiten interpretar mejor la existencia de ciertos residuos elevados en las predicciones.

---

**Pregunta 2.1** — Indica los valores de MAE, RMSE y R² de la regresión lineal sobre el test set. ¿El modelo funciona bien? ¿Por qué?

> En el conjunto de test, la regresión lineal ha obtenido los siguientes resultados: MAE = 731.5479, RMSE = 1102.1477 y R² = 0.9222.

En general, el modelo funciona bien porque explica el 92.22% de la variabilidad de la variable objetivo (`price`) en los datos de prueba, lo que indica una capacidad predictiva alta. El MAE muestra que, en promedio, el error absoluto de predicción es de unos 731.55 dólares, mientras que el RMSE asciende a 1102.15 dólares, lo que refleja que existen algunos errores más grandes penalizados con mayor intensidad.

No se aprecia un problema claro de underfitting, ya que el ajuste es elevado, ni se puede afirmar overfitting solo con estas métricas, porque la evaluación se ha realizado sobre el conjunto de test y el rendimiento sigue siendo alto. Las variables más influyentes del modelo son `carat`, varias categorías de `clarity` y `color`, además de la dimensión `x`. No obstante, la interpretación individual de algunos coeficientes debe hacerse con cautela debido a la multicolinealidad detectada en el Ejercicio 1 entre `carat`, `x`, `y` y `z`.


---

## Ejercicio 3 — Regresión Lineal Múltiple en NumPy

---
En este ejercicio se ha implementado desde cero una regresión lineal múltiple utilizando exclusivamente NumPy, sin recurrir a Scikit-Learn para el ajuste del modelo. A partir de la formulación matricial de mínimos cuadrados ordinarios (OLS), se ha añadido una columna de unos para estimar el intercepto, se han calculado los coeficientes del modelo y se han generado predicciones sobre el conjunto de test sintético incluido en el archivo base.

Los resultados obtenidos muestran que la implementación funciona correctamente. Los coeficientes ajustados han sido próximos a los valores reales usados para generar los datos, lo que indica que la solución OLS está bien programada. En concreto, se han estimado coeficientes cercanos a [5.0, 2.0, -1.0, 0.5], con pequeñas desviaciones atribuibles al ruido aleatorio incorporado al proceso de generación.

En cuanto a la evaluación, el modelo ha obtenido MAE = 1.1665, RMSE = 1.4612 y R² = 0.6897. El MAE y el RMSE se aproximan bien a los valores de referencia del enunciado, mientras que el R² ha quedado algo por debajo del valor orientativo esperado. Aun así, el comportamiento general del modelo es coherente y los resultados permiten validar que tanto la estimación de coeficientes como el cálculo de métricas y la generación del gráfico real vs. predicho se han implementado correctamente.

---

**Pregunta 3.1** — Explica en tus propias palabras qué hace la fórmula β = (XᵀX)⁻¹ Xᵀy y por qué es necesario añadir una columna de unos a la matriz X.

>La fórmula β = (XᵀX)⁻¹ Xᵀy calcula los coeficientes del modelo de regresión lineal múltiple mediante la solución analítica de mínimos cuadrados ordinarios (OLS). Su objetivo es encontrar los valores de β que minimizan la suma de los errores cuadrados entre los valores reales y los valores predichos por el modelo.

Es necesario añadir una columna de unos a la matriz X para poder estimar el intercepto o término independiente β₀. Sin esa columna, el modelo solo ajustaría los coeficientes de las variables predictoras y obligaría a que la recta o hiperplano pasara por el origen, lo que en general no representa correctamente la relación entre las variables.

**Pregunta 3.2** — Copia aquí los cuatro coeficientes ajustados por tu función y compáralos con los valores de referencia del enunciado.



> | Parametro | Valor real | Valor ajustado |
|-----------|-----------|----------------|
| β₀        | 5.0       | 4.8650         |
| β₁        | 2.0       | 2.0636         |
| β₂        | -1.0      | -1.1170        |
| β₃        | 0.5       | 0.4385         |

Los coeficientes ajustados son razonablemente próximos a los valores reales de referencia. El intercepto y los tres coeficientes presentan pequeñas desviaciones, pero en conjunto el ajuste reproduce correctamente la estructura del modelo generador de los datos. Esto indica que la implementación de la regresión lineal múltiple mediante OLS con NumPy es funcional y coherente.

**Pregunta 3.3** — ¿Qué valores de MAE, RMSE y R² has obtenido? ¿Se aproximan a los de referencia?

> > Los valores obtenidos han sido: MAE = 1.1665, RMSE = 1.4612 y R² = 0.6897.

El MAE y el RMSE se aproximan bien a los valores de referencia del enunciado, ya que están muy cerca de 1.20 y 1.50 respectivamente. El R² ha quedado por debajo del valor orientativo esperado, pero el ajuste sigue siendo razonable y consistente con unos coeficientes estimados cercanos a los reales. En conjunto, los resultados permiten considerar correcta la implementación.

**Pregunta 3.4* — Compara los resultados con la reacción logística anterior para tu dataset y comprueba si el resultado es parecido. Explica qué ha sucedido. 

> Interpretando esta pregunta como una comparación con el modelo de regresión del Ejercicio 2, los resultados no son directamente comparables en términos numéricos, porque ambos ejercicios trabajan con datos y contextos distintos. En el Ejercicio 2 se utilizó un dataset real de diamantes, con variables numéricas y categóricas, preprocesamiento con codificación y escalado, y el modelo obtuvo un R² de 0.9222. En cambio, en el Ejercicio 3 se trabaja con datos sintéticos generados en el propio archivo, con solo tres variables predictoras y un ajuste implementado manualmente con NumPy, obteniéndose un R² de 0.6897.

Aun así, ambos ejercicios sí son comparables desde el punto de vista conceptual, ya que en los dos casos se está resolviendo un problema de regresión lineal. La diferencia principal es que en el Ejercicio 2 se usa una herramienta de alto nivel como Scikit-Learn sobre un problema real, mientras que en el Ejercicio 3 se implementa manualmente la mecánica interna del ajuste OLS sobre un caso controlado. Por tanto, los resultados no tienen por qué ser parecidos en magnitud, pero sí muestran la misma lógica de modelado: estimar una variable continua a partir de variables predictoras y evaluar el ajuste con MAE, RMSE y R².

---

## Ejercicio 4 — Series Temporales
---
En este ejercicio se ha analizado una serie temporal sintética diaria generada con semilla fija, que cubre el periodo 2018-01-01 a 2023-12-31, con un total de 2191 observaciones. A partir de la visualización inicial y de la descomposición aditiva con periodo 365, se identifican con claridad una tendencia creciente, un patrón estacional anual, un ciclo de largo plazo y un componente residual de ruido.

La tendencia observada es aproximadamente lineal y ascendente. A lo largo de los seis años, el nivel de la serie aumenta en torno a 110 unidades, lo que equivale aproximadamente a un crecimiento de 0.05 unidades por día. Además, la componente estacional muestra un comportamiento recurrente con periodicidad anual, repitiéndose cada 365 días.

También se aprecia un ciclo de largo plazo, más suave y de frecuencia menor que la estacionalidad, con un periodo cercano a 4 años (≈1461 días). Este ciclo se distingue de la tendencia porque no mantiene un crecimiento constante, sino una oscilación lenta de subida y bajada superpuesta al crecimiento global.

En cuanto al residuo, los resultados indican un comportamiento muy próximo al de un ruido ideal. La media del residuo es 0.127078, la desviación típica 3.222043, la asimetría -0.050917 y la curtosis -0.061028, valores muy cercanos a los esperables en un ruido aproximadamente gaussiano y centrado. El test de Jarque-Bera no rechaza la normalidad (p-value = 0.576561), mientras que el test ADF confirma estacionariedad del residuo (p-value = 0.000000). En conjunto, el componente residual puede considerarse compatible con un ruido aproximadamente normal, estacionario y sin estructura sistemática dominante.

---

**Pregunta 4.1** — ¿La serie presenta tendencia? Descríbela brevemente (tipo, dirección, magnitud aproximada).

> Sí, la serie presenta una tendencia claramente creciente y de tipo aproximadamente lineal. A lo largo del periodo completo, el nivel de la serie aumenta en torno a 110 unidades, lo que equivale aproximadamente a una pendiente media de 0.05 unidades por día. Esta tendencia refleja un crecimiento sostenido del nivel base de la serie a lo largo del tiempo.

**Pregunta 4.2** — ¿Hay estacionalidad? Indica el periodo aproximado en días y la amplitud del patrón estacional.

> Sí, existe una estacionalidad clara con un periodo aproximado de 365 días, es decir, anual. El patrón estacional se repite de forma regular cada año y presenta una amplitud aproximada de 32 unidades pico a pico, lo que equivale a unas oscilaciones de alrededor de ±16 unidades respecto al nivel medio estacional.

**Pregunta 4.3** — ¿Se aprecian ciclos de largo plazo en la serie? ¿Cómo los diferencias de la tendencia?

> Sí, se aprecian ciclos de largo plazo además de la tendencia. Su periodo aproximado es de unos 1461 días, es decir, cerca de 4 años. Se diferencian de la tendencia porque la tendencia representa un crecimiento global sostenido y aproximadamente lineal, mientras que el ciclo de largo plazo introduce oscilaciones lentas de subida y bajada alrededor de esa trayectoria creciente.

**Pregunta 4.4** — ¿El residuo se ajusta a un ruido ideal? Indica la media, la desviación típica y el resultado del test de normalidad (p-value) para justificar tu respuesta.

> > El residuo se ajusta razonablemente bien a un ruido ideal. La media obtenida es 0.127078 y la desviación típica 3.222043, lo que indica que el residuo está centrado cerca de cero y tiene una dispersión moderada. Además, la asimetría (-0.050917) y la curtosis (-0.061028) son muy próximas a cero, lo que apunta a una forma cercana a la normal.

En el test de normalidad Jarque-Bera se obtiene un p-value de 0.576561, por lo que no se rechaza la hipótesis de normalidad. Por su parte, el test ADF devuelve un p-value de 0.000000, lo que indica que el residuo es estacionario. En conjunto, estos resultados apoyan que el residuo puede interpretarse como un ruido aproximadamente gaussiano, centrado y sin tendencia.

---

*Fin del documento de respuestas*
