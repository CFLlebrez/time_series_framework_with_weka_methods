# Documentación de Métodos de Filtro de scikit-learn

## Descripción General

Este módulo extiende el framework de selección de atributos para series temporales con métodos de filtro de scikit-learn. Estos métodos son computacionalmente eficientes y no dependen de un modelo específico, lo que los hace particularmente útiles para el análisis de series temporales.

## Métodos de Filtro Implementados

### SelectKBest
- **Descripción**: Selecciona las k mejores características según una función de puntuación.
- **Funciones de puntuación**:
  - `f_regression`: Para variables numéricas continuas, mide la correlación lineal.
  - `mutual_info_regression`: Captura relaciones no lineales entre variables.
- **Parámetros clave**:
  - `n_features`: Número de características a seleccionar.
  - `score_func_name`: Nombre de la función de puntuación ('f_regression' o 'mutual_info_regression').

### SelectPercentile
- **Descripción**: Selecciona un porcentaje de características en lugar de un número fijo.
- **Parámetros clave**:
  - `percentile`: Porcentaje de características a seleccionar (0-100).
  - `score_func_name`: Nombre de la función de puntuación.

### GenericUnivariateSelect
- **Descripción**: Permite seleccionar características usando diferentes estrategias estadísticas.
- **Estrategias**:
  - `k_best`: Selecciona las k mejores características.
  - `percentile`: Selecciona un porcentaje de características.
  - `fpr`: Controla la tasa de falsos positivos.
  - `fdr`: Controla la tasa de falsos descubrimientos.
  - `fwe`: Controla la tasa de error familiar.
- **Parámetros clave**:
  - `strategy`: Estrategia de selección.
  - `param`: Parámetro específico para la estrategia elegida.
  - `score_func_name`: Nombre de la función de puntuación.

### VarianceThreshold
- **Descripción**: Elimina características con varianza por debajo de un umbral.
- **Parámetros clave**:
  - `threshold`: Umbral de varianza para la selección.

## Uso Básico

```python
from feature_selection import select_features

# Usando SelectKBest con f_regression
results = select_features(
    input_file='data.csv',
    output_dir='results/selectkbest',
    target_col='temperatura',
    method='sklearn_filter',
    max_lag=5,
    method_params='selectkbest',
    n_features=10,
    score_func_name='f_regression'
)

# Usando SelectPercentile
results = select_features(
    input_file='data.csv',
    output_dir='results/selectpercentile',
    target_col='temperatura',
    method='sklearn_filter',
    max_lag=5,
    method_params='selectpercentile',
    percentile=30,
    score_func_name='f_regression'
)

# Usando GenericUnivariateSelect
results = select_features(
    input_file='data.csv',
    output_dir='results/generic',
    target_col='temperatura',
    method='sklearn_filter',
    max_lag=5,
    method_params='genericunivariateselect',
    strategy='k_best',
    param=10,
    score_func_name='f_regression'
)

# Usando VarianceThreshold
results = select_features(
    input_file='data.csv',
    output_dir='results/variance',
    target_col='temperatura',
    method='sklearn_filter',
    max_lag=5,
    method_params='variancethreshold',
    threshold=0.1
)
```

## Ejemplos Avanzados

### Selección de características con información mutua

```python
results = select_features(
    input_file='data.csv',
    output_dir='results/mutual_info',
    target_col='temperatura',
    method='sklearn_filter',
    max_lag=5,
    method_params='selectkbest',
    n_features=10,
    score_func_name='mutual_info_regression'
)
```

### Control de falsos descubrimientos

```python
results = select_features(
    input_file='data.csv',
    output_dir='results/fdr',
    target_col='temperatura',
    method='sklearn_filter',
    max_lag=5,
    method_params='genericunivariateselect',
    strategy='fdr',
    param=0.05,
    score_func_name='f_regression'
)
```

## Integración con el Framework

Los métodos de filtro de scikit-learn se integran perfectamente con el framework existente:

1. **Misma interfaz**: Utilizan la misma interfaz que los otros métodos de selección.
2. **Mismas salidas**: Generan los mismos tipos de salidas (listado de características, puntuaciones, visualizaciones, CSV filtrado).
3. **Compatibilidad**: Pueden combinarse con los otros métodos en flujos de trabajo.

## Ventajas de los Métodos de Filtro

- **Eficiencia computacional**: Son más rápidos que los métodos basados en modelos.
- **Independencia del modelo**: No dependen de un algoritmo de aprendizaje específico.
- **Escalabilidad**: Funcionan bien con conjuntos de datos grandes.
- **Interpretabilidad**: Proporcionan puntuaciones directamente interpretables.

## Limitaciones

- **Univariados**: La mayoría de estos métodos evalúan cada característica de forma independiente, sin considerar interacciones.
- **Lineales**: Algunos métodos (como f_regression) asumen relaciones lineales.
- **No específicos para series temporales**: No consideran la naturaleza temporal de los datos directamente.

## Prueba de los Métodos

El script `test_sklearn_filter.py` demuestra el uso de los diferentes métodos de filtro con datos de ejemplo y genera informes comparativos.

```bash
python test_sklearn_filter.py
```

## Requisitos

- scikit-learn >= 0.24.0
- pandas
- numpy
- matplotlib
- seaborn
