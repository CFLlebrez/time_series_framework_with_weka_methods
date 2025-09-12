# Métodos de Selección de Atributos Inspirados en Weka

Este documento describe los métodos de selección de atributos inspirados en Weka implementados en el framework de series temporales.

## Introducción

Los métodos de selección de atributos son fundamentales para mejorar el rendimiento de los modelos predictivos, especialmente en series temporales multivariables donde puede haber muchas variables y lags a considerar. Los métodos inspirados en Weka implementados en este framework ofrecen enfoques potentes y complementarios a los métodos ya existentes.

## Métodos Implementados

### 1. CFS (Correlation-based Feature Selection)

**Descripción**: CFS evalúa subconjuntos de atributos en lugar de atributos individuales. Selecciona subconjuntos que tienen alta correlación con la clase pero baja correlación entre ellos.

**Principio**: "Buenos conjuntos de características contienen características altamente correlacionadas con la clase, pero no correlacionadas entre sí".

**Fórmula**:
```
Merit = (k * rcf) / sqrt(k + k * (k-1) * rff)
```
Donde:
- k: número de características en el subconjunto
- rcf: correlación media característica-clase
- rff: correlación media característica-característica

**Algoritmo de búsqueda**: Utiliza BestFirst para explorar el espacio de posibles subconjuntos de atributos.

**Parámetros principales**:
- `n_features`: Número máximo de características a seleccionar
- `max_backtrack`: Número máximo de retrocesos en la búsqueda BestFirst

### 2. InfoGain (Information Gain)

**Descripción**: InfoGain evalúa atributos midiendo su ganancia de información con respecto a la clase. Calcula cuánta información proporciona cada atributo sobre la clase, basándose en el concepto de entropía.

**Fórmula**:
```
IG(D, A) = H(D) - H(D|A)
```
Donde:
- H(D): entropía del conjunto de datos
- H(D|A): entropía condicional (entropía de D dado A)

**Parámetros principales**:
- `n_features`: Número de características a seleccionar
- `discretize`: Si es True, discretiza variables continuas
- `n_bins`: Número de bins para discretización

### 3. ReliefF

**Descripción**: ReliefF evalúa la calidad de los atributos basándose en cómo distinguen entre instancias cercanas. Es especialmente útil para detectar interacciones entre atributos.

**Principio**: Penaliza las diferencias de atributos entre instancias cercanas de la misma clase y recompensa las diferencias entre instancias de diferentes clases.

**Variantes**:
- ReliefF: Para problemas de clasificación
- RReliefF: Para problemas de regresión (implementado automáticamente cuando la variable objetivo es continua)

**Parámetros principales**:
- `n_features`: Número de características a seleccionar
- `n_neighbors`: Número de vecinos a considerar
- `sample_size`: Tamaño de la muestra a usar (None = usar todos)

## Uso

### Uso Básico

```python
from feature_selection import select_features

# Usando CFS
results = select_features(
    input_file='data.csv',
    output_dir='results',
    target_col='temperatura',
    method='weka_inspired',
    method='cfs',
    n_features=10,
    max_backtrack=5
)

# Usando InfoGain
results = select_features(
    input_file='data.csv',
    output_dir='results',
    target_col='temperatura',
    method='weka_inspired',
    method='infogain',
    n_features=10,
    discretize=True,
    n_bins=10
)

# Usando ReliefF
results = select_features(
    input_file='data.csv',
    output_dir='results',
    target_col='temperatura',
    method='weka_inspired',
    method='relieff',
    n_features=10,
    n_neighbors=10
)
```

### Uso desde Línea de Comandos

```bash
python time_series_framework.py input.csv output_dir --fv 1 --fh 3 --ph 5 --feature_selection --fs_method weka_inspired --method cfs --n_features 10
```

## Resultados

Todos los métodos generan los siguientes resultados:

1. **Lista de características seleccionadas**: Un listado de las variables y lags seleccionados.
2. **Puntuaciones de importancia**: Valores numéricos que indican la importancia de cada característica.
3. **Visualizaciones**: Gráficos que muestran la importancia relativa de las características.
4. **CSV filtrado**: Un archivo CSV que contiene solo las variables y lags seleccionados.

## Comparación con Otros Métodos

### Ventajas de CFS
- Considera interacciones entre características
- Reduce la redundancia en el conjunto seleccionado
- No requiere establecer un número fijo de características

### Ventajas de InfoGain
- Rápido y eficiente computacionalmente
- Fácil de interpretar
- Funciona bien con datos categóricos y numéricos

### Ventajas de ReliefF
- Detecta interacciones complejas entre características
- Robusto frente a datos ruidosos
- Maneja bien datos con múltiples clases

## Referencias

1. Hall, M. A. (1999). Correlation-based Feature Selection for Machine Learning. The University of Waikato.
2. Kononenko, I. (1994). Estimating attributes: Analysis and extensions of RELIEF. European Conference on Machine Learning.
3. Quinlan, J. R. (1986). Induction of decision trees. Machine Learning, 1(1), 81-106.
