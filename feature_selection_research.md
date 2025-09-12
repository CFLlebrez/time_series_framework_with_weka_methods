# Métodos de Selección de Atributos para Series Temporales

## Métodos basados en correlación

### Correlación de Pearson
- Mide la relación lineal entre variables
- Rango: -1 a 1 (correlación negativa perfecta a correlación positiva perfecta)
- Implementación: `pandas.corr()`, `numpy.corrcoef()`, `scipy.stats.pearsonr()`
- Ventajas: Simple, interpretable
- Desventajas: Solo captura relaciones lineales

### Análisis de correlación cruzada (CCF)
- Mide la correlación entre series temporales con diferentes lags
- Implementación: `pandas.DataFrame.rolling_corr()`, `statsmodels.tsa.stattools.ccf()`
- Ventajas: Captura relaciones temporales
- Desventajas: Computacionalmente intensivo para muchas variables y lags

### Información mutua
- Mide la dependencia no lineal entre variables
- Implementación: `sklearn.feature_selection.mutual_info_regression()`, `scipy.stats.mutual_info_score()`
- Ventajas: Captura relaciones no lineales
- Desventajas: Requiere estimación de densidad, sensible al tamaño de muestra

## Métodos basados en modelos

### Feature importance de Random Forest
- Mide la importancia de cada variable en un modelo de Random Forest
- Implementación: `sklearn.ensemble.RandomForestRegressor().feature_importances_`
- Ventajas: Captura interacciones no lineales, robusto a outliers
- Desventajas: Puede ser sesgado hacia variables con muchas categorías

### Lasso (L1) y Elastic Net
- Regularización que puede llevar coeficientes a cero, efectivamente seleccionando variables
- Implementación: `sklearn.linear_model.Lasso()`, `sklearn.linear_model.ElasticNet()`
- Ventajas: Integra selección y modelado, maneja multicolinealidad
- Desventajas: Sensible a la escala de las variables

### Recursive Feature Elimination (RFE)
- Elimina recursivamente las variables menos importantes
- Implementación: `sklearn.feature_selection.RFE()`
- Ventajas: Considera interacciones entre variables
- Desventajas: Computacionalmente intensivo, puede ser inestable

## Métodos específicos para series temporales

### Análisis de causalidad de Granger
- Prueba si una serie temporal ayuda a predecir otra
- Implementación: `statsmodels.tsa.stattools.grangercausalitytests()`
- Ventajas: Específico para relaciones temporales causales
- Desventajas: Asume estacionariedad, sensible a la especificación del modelo

### Análisis de componentes principales (PCA)
- Reduce dimensionalidad preservando la varianza
- Implementación: `sklearn.decomposition.PCA()`
- Ventajas: Reduce multicolinealidad, puede mejorar rendimiento
- Desventajas: Componentes pueden ser difíciles de interpretar

### Análisis espectral
- Identifica patrones cíclicos y estacionales
- Implementación: `scipy.signal.periodogram()`, `statsmodels.tsa.stattools.acf()`
- Ventajas: Captura patrones periódicos
- Desventajas: Requiere series estacionarias, sensible a ruido

## Métodos de selección automática

### Selección secuencial hacia adelante/atrás (SFS/SBS)
- Construye conjuntos óptimos de variables iterativamente
- Implementación: `mlxtend.feature_selection.SequentialFeatureSelector()`
- Ventajas: Simple, interpretable
- Desventajas: Puede quedar atrapado en óptimos locales

### Algoritmos genéticos
- Explora combinaciones de variables y lags usando principios evolutivos
- Implementación: `DEAP`, `PyGAD`
- Ventajas: Explora espacio de búsqueda amplio
- Desventajas: Computacionalmente intensivo, requiere ajuste de hiperparámetros

## Bibliotecas necesarias
- pandas
- numpy
- scipy
- scikit-learn
- statsmodels
- mlxtend
- DEAP o PyGAD (para algoritmos genéticos)
