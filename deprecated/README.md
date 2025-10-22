# Documentación del Módulo de Selección de Atributos

## Descripción General

El módulo de selección de atributos para series temporales proporciona herramientas para identificar las variables y lags más relevantes para la predicción de una variable objetivo. Este módulo se integra con el framework existente de series temporales y ofrece múltiples métodos de selección, visualizaciones avanzadas y generación de informes detallados.

## Métodos de Selección Implementados

### Métodos basados en correlación
- **Correlación de Pearson**: Selecciona atributos basados en su correlación lineal con la variable objetivo.
- **Análisis de correlación cruzada (CCF)**: Identifica relaciones con diferentes lags temporales.
- **Información mutua**: Captura relaciones no lineales entre variables.

### Métodos basados en modelos
- **Feature importance de Random Forest**: Utiliza la importancia de variables de un modelo Random Forest.
- **Lasso (L1)**: Selecciona variables mediante regularización L1.
- **Elastic Net**: Combina regularización L1 y L2 para selección de variables.
- **Recursive Feature Elimination (RFE)**: Elimina recursivamente las variables menos importantes.

### Métodos específicos para series temporales
- **Análisis de causalidad de Granger**: Identifica si una serie temporal ayuda a predecir otra.
- **Análisis de componentes principales (PCA)**: Reduce dimensionalidad preservando la varianza.
- **Análisis espectral**: Identifica patrones cíclicos y estacionales.

### Métodos de selección automática
- **Selección secuencial hacia adelante/atrás (SFS/SBS)**: Construye conjuntos óptimos de variables.
- **Algoritmos genéticos**: Explora combinaciones de variables y lags.

## Estructura del Módulo

```
feature_selection/
├── __init__.py                  # Interfaz principal y funciones de alto nivel
├── correlation_based.py         # Métodos basados en correlación
├── model_based.py               # Métodos basados en modelos
├── time_series_specific.py      # Métodos específicos para series temporales
├── automatic_selection.py       # Métodos de selección automática
└── visualization.py             # Funciones para visualizar resultados
```

## Uso Básico

```python
from feature_selection import select_features

# Seleccionar características usando Random Forest
results = select_features(
    input_file='data.csv',
    output_dir='results',
    target_col='temperatura',
    method='random_forest',
    n_features=10,
    max_lag=5
)

# Obtener características seleccionadas
selected_features = results['selected_features']

# Obtener importancias
importances = results['feature_importances']

# Rutas a archivos generados
report_path = results['report_path']
plot_path = results['plot_path']
filtered_csv_path = results['filtered_csv_path']
```

## Parámetros Principales

- **input_file**: Ruta al archivo CSV de entrada con datos de series temporales.
- **output_dir**: Directorio para guardar resultados (informes, visualizaciones, CSV filtrado).
- **target_col**: Nombre de la columna objetivo a predecir.
- **method**: Método de selección de atributos a utilizar.
- **n_features**: Número de características a seleccionar.
- **threshold**: Umbral alternativo para selección de características.
- **max_lag**: Máximo lag a considerar para las variables.

## Salidas Generadas

1. **Listado de características seleccionadas**: Lista ordenada por importancia.
2. **Puntuaciones de importancia**: Valor numérico que indica la relevancia de cada característica.
3. **Visualizaciones**:
   - Gráfico de barras de importancia de características
   - Matriz de correlación
   - Distribución de características seleccionadas
   - Importancia por lag para cada variable
   - Gráfico de coordenadas paralelas
4. **CSV filtrado**: Archivo con solo las variables y lags seleccionados.
5. **Informe detallado**: Archivo CSV con todas las características y sus métricas.

## Ejemplos de Uso

### Selección basada en correlación cruzada

```python
results = select_features(
    input_file='data.csv',
    output_dir='results/ccf',
    target_col='temperatura',
    method='ccf',
    n_features=15,
    max_lag=10
)
```

### Selección basada en Lasso

```python
results = select_features(
    input_file='data.csv',
    output_dir='results/lasso',
    target_col='temperatura',
    method='lasso',
    alpha=0.01,
    threshold=0.001
)
```

### Selección basada en causalidad de Granger

```python
results = select_features(
    input_file='data.csv',
    output_dir='results/granger',
    target_col='temperatura',
    method='granger',
    max_lag=8,
    threshold=0.05
)
```

### Selección secuencial

```python
results = select_features(
    input_file='data.csv',
    output_dir='results/sequential',
    target_col='temperatura',
    method='sequential',
    n_features=10,
    direction='forward',
    scoring='neg_mean_squared_error'
)
```

## Integración con el Framework de Series Temporales

El módulo de selección de atributos se integra perfectamente con el framework existente de series temporales:

1. **Preprocesamiento**: Utilice la selección de atributos antes de la transformación para identificar las variables y lags más relevantes.

2. **Transformación filtrada**: El CSV filtrado generado puede utilizarse como entrada para el script de transformación, reduciendo la dimensionalidad y mejorando el rendimiento.

3. **Análisis exploratorio**: Las visualizaciones generadas proporcionan insights valiosos sobre las relaciones entre variables y la importancia de diferentes lags.

## Prueba del Módulo

El script `test_feature_selection.py` demuestra el uso de los diferentes métodos de selección con datos de ejemplo y genera informes comparativos.

```bash
python test_feature_selection.py
```

## Requisitos

- pandas
- numpy
- scikit-learn
- statsmodels
- matplotlib
- seaborn
- tqdm

## Notas Importantes

- Para conjuntos de datos grandes, los métodos basados en modelos y la selección automática pueden ser computacionalmente intensivos.
- El método genético es particularmente exigente y debe usarse con precaución.
- Para series temporales con estacionalidad, considere utilizar el análisis espectral o CCF.
- La selección de atributos debe validarse con datos de prueba para asegurar que mejora el rendimiento predictivo.
