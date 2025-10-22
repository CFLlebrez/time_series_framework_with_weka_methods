# Framework para Series Temporales Multivariables

Este framework proporciona herramientas para trabajar con series temporales multivariables en tareas de predicción. El componente principal es un script de Python que transforma datos de series temporales en un formato adecuado para regresión multivariable.

## Descripción

El script `time_series_transformer.py` (y su versión optimizada `time_series_transformer_optimized.py`) transforma un archivo CSV que contiene datos de series temporales multivariables en un formato adecuado para análisis de regresión. Toma una serie temporal donde cada fila es un registro temporal con múltiples variables y la transforma en un formato donde cada fila contiene:

- Valores pasados de todas las variables (para contexto)
- Valores futuros de la variable objetivo (para predicción)

## Parámetros

El script acepta los siguientes parámetros:

- `input_file`: Ruta al archivo CSV de entrada
- `output_file`: Ruta al archivo CSV de salida
- `--fv`: Variable de Pronóstico (Forecast Variable) - índice de la columna a predecir
- `--fh`: Horizonte de Pronóstico (Forecast Horizon) - número de valores futuros a predecir
- `--ph`: Historial Pasado (Past History) - número de valores pasados a utilizar para la predicción

## Formato de Entrada

El archivo CSV de entrada debe tener el siguiente formato:
- Cada fila representa un registro temporal
- Cada columna representa una variable diferente
- La primera fila debe contener los nombres de las columnas

Ejemplo:
```
fecha,temperatura,humedad,presion,velocidad_viento
2023-01-01,22.5,65.3,1013.2,12.4
2023-01-02,23.1,63.7,1012.8,10.9
...
```

## Formato de Salida

El archivo CSV de salida tendrá el siguiente formato:
- Cada fila representa una muestra para entrenamiento/predicción
- Las primeras columnas contienen los valores pasados de todas las variables
- Las últimas columnas contienen los valores futuros de la variable objetivo

Los nombres de las columnas siguen el formato:
- Para valores pasados: `{nombre_variable}_t-{paso_tiempo}`
- Para valores futuros: `{nombre_variable_objetivo}_t+{paso_tiempo}`

Ejemplo (con FV=1, FH=2, PH=3):
```
fecha_t-3,temperatura_t-3,humedad_t-3,...,fecha_t-1,temperatura_t-1,humedad_t-1,...,temperatura_t+1,temperatura_t+2
```

## Instalación

El script requiere Python 3.6+ y las siguientes dependencias:
- pandas
- numpy
- tqdm (solo para la versión optimizada)

Instale las dependencias con:
```
pip install pandas numpy tqdm
```

## Uso

### Versión Básica

```bash
python time_series_transformer.py input.csv output.csv --fv 1 --fh 3 --ph 5
```

### Versión Optimizada

```bash
python time_series_transformer_optimized.py input.csv output.csv --fv 1 --fh 3 --ph 5
```

## Ejemplos

### Predecir temperatura con 5 valores pasados y 3 valores futuros

```bash
python time_series_transformer.py sample_data.csv output.csv --fv 1 --fh 3 --ph 5
```

### Predecir humedad con 4 valores pasados y 2 valores futuros

```bash
python time_series_transformer.py sample_data.csv output.csv --fv 2 --fh 2 --ph 4
```

## Notas Importantes

- El índice FV comienza en 0, por lo que FV=1 se refiere a la segunda columna del archivo CSV.
- El número total de muestras generadas será: `número_de_filas_entrada - (PH + FH) + 1`
- Asegúrese de que su archivo CSV tenga suficientes filas para generar al menos una muestra (mínimo PH + FH filas).

## Limitaciones

- El script actual no maneja valores faltantes (NaN). Asegúrese de que su archivo CSV no contenga valores faltantes.
- Para conjuntos de datos muy grandes, considere utilizar la versión optimizada que incluye una barra de progreso.
