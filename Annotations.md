## 14/09/2025 ##
- Reestructurado el repositorio: utiliza solo time_series_transformer_optimized.py.
- Creado gather_csv.py para obtener csv de API's (Por ahora del RIA).

- **OBJETIVO**: tomar csv de datos temporales -> csv procesado con lags hasta ph (Past History), valores futuros hasta fh (Forecast Horizon) para fv (Forecast Variable).

- Mejorado el sistema de lectura y escritura de ficheros (añade informacion de los parametros en el nombre del fichero de salida).

- Entendido el fichero del transformer: **El transformer de preprocesamiento funciona correctamente.**: en cada i (entre 0 y total-(fh+ph)) toma valores pasados y valores futuros de acuerdo con los parametros. Los datos transformados los guarda en filas. Crea los nombres de las columnas y crea el nuevo DataFrame

### Framework ### 
- Creada nueva carpeta para resultados (habrá que optimizar pero cuando funcione).
- Creado requirements.txt
- Intentado crear entorno virtual para reducir la version de numpy (unauthorized)
- Comando que trataba de ejecutar el framework: python time_series_framework.py estacion.csv ria1_results --fv 8 --fh 4 --ph 4 --feature_selection 1 --fs_method lasso --fs_n_features 2
- Ultimo resultado: No module named 'statsmodels'.

## 16/09/2025 ##
- Instalado y añadido a requirements statsmodels
- Corregido comando: python time_series_framework.py estacion.csv ria1_results --fv 8 --fh 4 --ph 4 --feature_selection --fs_method lasso --fs_n_features 2

- Problema actual -> la funcion prepare_lagged_data devuelve un dataframe vacío. **Revisar las distintas funciones para selección de atributos para pensar si integrar directamente la funcion de time_series_transformer_optimized y utilizar el fichero de salida de ese programa para llamar al resto de funciones.** i.e. Ver donde se utiliza X_lagged y repasar esas funciones.
- X_lagged es un df.
- y es un array con la variable objetivo alineada con X_lagged.

- Transformer devuelve un unico df con todos los lags de las variables predictoras, y valores futuros de la variable a predecir.

- **Razón del problema**: X_lagged tiene NaN en todas las filas por tanto al hacer dropna elimina todo. La columna et0 estaba vacía, he probado a eliminarla para probar así.
- **Solución**: La columna et0 estaba vacía, al borrarla olvidé eliminar la coma del final, eso estaba haciendo que se rompiese la lectura. Está solucionado, va funcionando. 

- **Siguiente problema**: Ha ejecutado la selección, ha generado el CSV filtrado (*revisar* results/ria1_results\feature_selection\filtered_data_random_forest.csv, puede que el problema sean las "\", no ha encontrado el fichero para leer para transformar la serie temporal).
- **Solución**: Revisar para que todas las direcciones sean coherentes.

