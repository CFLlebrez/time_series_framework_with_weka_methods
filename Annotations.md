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

- **Siguiente problema**: Ha ejecutado la selección, ha generado el CSV filtrado (*revisar* results/ria1_results\feature_selection\filtered_data_random_forest.csv, puede que el problema sean las "\\", no ha encontrado el fichero para leer para transformar la serie temporal).
- **Solución**: Revisar para que todas las direcciones sean coherentes.

## 19/09/2025 ##
- Cambiado el formato de las rutas de ficheros para que fuese consistente con los previos.
- Actualmente hace todo correcto menos: Si ha ejecutado la selección de atributos, el csv que genera guardando las características relevantes no contiene la variable objetivo.
- Lo que hay que cambiar: 1. Si hay selección no hace falta transformación. 2. Añadir variable objetivo al csv filtrado.

- Cambios: 
1. Guarda la fecha en caso de que la primera columna sea fecha para mantenerla en el filtrado (cambios en __init__.py)
2. Guarda la variable objetivo en la última columna y la pasa correctamente al transformer.
3. En el transformer se ha añadido un parametro original_fv para mejorar los ficheros creados (ver en el nombre si ha utilizado seleccion de atributos y qué parámetro se pasó para la variable objetivo i.e. su indice en el csv original)
4. Para añadir flag de selección de atributos actualmente es pobre, cambiar por pasar un boolean en vez de comparar fv con original_fv.

- Conclusión del día: Funciona correctamente con el comando anterior. Cosas que probar:
1. Otros parámetros para el mismo csv.
2. Otros parámetros de selección de atributos.
3. Mirar los ficheros de selección de atributos en detalle para comprobar cómo funcionan.
(cambiar el método de adición del flag de selección de atributos y ver qué cambiar para que se pueda utilizar el transformer de forma independiente al framework tras haber añadido parámetros).
4. Intentar ver por qué al hacer el filtrado pierde 1 fila (con el lag 3 necesita desprenderse de 3 filas, pero empieza en la 5ª fecha en vez de en la 4ª)

## 20/09/2025 ##
- Observaciones: El filtrado pierde una fila efectivamente (aparte de las perdidas en función de máx lag por defecto 4). Al transformar el filtrado se pierden más (correspondientes a ph para hacer los lags especificados).
- Borrado el test inicial hecho con random_forest.
- Obtenido un csv más amplio para tener más libertad a la hora de obtener parámetros. (Otra vez manualmente he quitado la ultima columna, seguro que se puede quitar utilizando un rango).

- Cambio de workflow: Actualmente hacía primero seleccion de atributos y posteriormente transformación. La idea es hacer primero una transformación para obtener todas las variables sobre las que seleccionar y posteriormente hacer la selección.

- Fin del dia: ultimo comando python time_series_framework.py estacion_amplio.csv ria2_results --fv 8 --fh 6 --ph 6 --feature_selection --fs_method lasso --fs_n_features 2
1. Ahora mismo hay varios problemas: lectura de ficheros, uso de insert (ejecutar), buscar la forma de preservar la fv utilizada desde el principio y que no cambie a la mitad.
2. Hay que revisar todos los lugares donde se escriben/leen ficheros para hacer el programa consistente con los nombres.
3. Ver qué es lo de insert.
4. Revisar el uso de original_fv (probablemente ya no sea necesario o cambiar para utilizarlo más adelante)

## 21/09/2025 ##
- Revisados framework.py, transformer.py e \_\_init__.py para ajustar los nombres de ficheros.
- Revisada mejor forma de mantener la fecha sin hacerle transformación y alineada con los datos filtrados: en el comando se introduce el nombre de la columna de fecha para utilizarla en los distintos metodos.
- Revisar la forma de tratar con la fecha en cada paso.

- Conclusión del día: 
1. El código funciona y parece que genera el csv correctamente.
2. El problema ahora está en la selección de características: ha seleccionado dos caracteristicas pero solo una tiene importancia, todas las demás tienen importancia 0.
3. Ver por qué está haciendo esa selección y si con otros métodos ocurre lo mismo.