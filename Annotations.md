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

## 22/09/2025 ##
- Al probar otros métodos y preguntar a gpt me ha aclarado que Lasso penaliza las no seleccionadas para llevarlas a 0, se queda únicamente con las características significativas (resto a 0) y solo con el lag más significativo (resto a 0).
- Random forest cuenta la variable objetivo como una de las características.
- Spectral devuelve orden distinto al esperado.
- Probados Lasso, Random_forest, Spectral.
- Probar 'pearson', 'ccf', 'mutual_info', 'elastic_net', 'rfe', 'granger', 'pca', 'sequential', 'genetic'.
- Pearson, CCF, Mutual_info, Elastic_net, RFE devuelven humedad max y su primer lag (arreglar la variable objetivo).
- Granger no funciona dado que utiliza max_lag (y tiene que ser estrictamente positivo).
- PCA devuelve tambien orden distinto al esperado (parece que PCA y Spectral proceden de forma similar y penalizan demasiada correlación con la variable objetivo).
- Sequential no funciona -> todas las variables importancia 0 y ningun error aparente.
- Genetic devuelve tantas características como encuentre, y utiliza todas las columnas incluidas variable objetivo y valores futuros. El problema es que trabaja muy lento, no se si se podrán ajustar los parámetros de población y generaciones (ha usado 50 y 20).

- Resuelto el problema al preparar X e y para predicción. 
- En principio se han eliminado las filas 'fecha', 'fv' y 'fv_t+i' para incluirse como características posibles. Seguramente los instantes futuros se utilicen para entrenar modelos como fv alternativos o multioutput models.

- Solucionar los problemas con Sequential y Granger.
- Granger solucionado únicamente forzando max_lag a 1 para no dar problemas (sigue dando warning de verbose)
- Sequential. Es probable que no funcione correctamente debido a que no hay secuencialidad ni periodicidad en los datos (periodo muy corto).

- PCA y Spectral miden otras características de la serie, no su potencial para predecir la variable objetivo.

- Conclusión del día: Probados todos los métodos incluidos en los parámetros de la función. Probar el resto de parámetros y otros csv (uno relativamente grande para aplicar Sequential correctamente).

- Varios de los métodos no se han ejecutado tras separar las variables objetivo y valores futuros de las características sobre las que seleccionar.

## 23/09/2025 ##
- Cambiado sequential que devolvía 0: en _evaluate_feature devolvía negativo para trabajar con neg_MSE y ahora positivo para trabajar con MSE (ahora devuelve valores demasiado grandes, lo corregiré).

- Ahora que el framework funciona mi siguiente objetivo es leer cada método detenidamente y entender los distintos aspectos que influyen (opinión pre-reunión).

- Resumen de qué decir en la reunión:
1. Empecé a trabajar con el framework, reestructuré algunas carpetas y corregí parte del código para que fuese consistente.
2. He estado cambiando distintas cosas para cambiar el flujo de trabajo: antes hacía selección de atributos y posteriormente transformaba el csv. Ahora primero lo transforma y realiza la selección directamente sobre el csv transformado (en el csv final no incluye los instantes futuros de la variable objetivo, solo las caracteristicas seleccionadas y la variable objetivo).
3. Una vez conseguí que funcionara correctamente para el csv que conseguí como muestra probé con otros y siempre que respete ciertos formatos (una columna de fecha) funciona correctamente.
4. Aún tengo que probar con csv's con muchos valores vacíos, porque en principio simplemente elimina las filas correspondientes pero se podría mejorar.
**(^)Hasta ahora/De ahora en adelante(v)**
5. Ahora lo que me dedicaré es a analizar más detenidamente los métodos para asegurarme de que no solo dan resultados con sentido sino que su implementación también es correcta (y tratar de entender la teoría que hay). También hay varias secciones de código que están desfasadas y dan warnings a pesar de que funcione.
6. En principio una vez haya hecho eso probaré con los métodos de weka porque son independientes del resto de métodos.
7. También trataré de añadir una documentación (actualizar la que generó manus).

## 24/09/2025 ##
- Reflexión: desde un principio valía con haber mantenido el flujo normal únicamente cambiando.
1. Estructura de carpetas para ficheros de entrada salida.
2. Separar variable objetivo (a menos que se quiera incluir) y fecha para realizar selección de atributos (con max_lag=ph).
3. Realizar la transformación añadiendo los lags y steps de la variable objetivo.
4. En caso de no haber selección de atributos se realiza la transformación sobre los datos originales sin la fecha.

- Próximo: Analizar los distintos métodos de selección de atributos y comprobar que haber cambiado el flujo inicial de trabajo no afecta a su correcto funcionamiento.
- Ahora mismo el csv filtrado solo necesita los instantes futuros que se predirán (y eliminar la variable objetivo en el instante concreto).

## 25/09/2025 ##
- Analizados \_\_init__.py y correlation_based.py
- Siguientes automatic_selection.py y model_based.py

## 01/10/2025 ##
- REUNION: He estado trabajando con el framework y he hecho los siguientes cambios/avances
1. Cambié un poco la estructura de carpetas.
2. Cambié el flujo de trabajo: antes filtraba y después transformaba lo filtrado, ahora primero transforma y después filtra sobre el csv transformado.
3. También lo arreglé para que no incluyese la columna de fecha ni la variable objetivo ni pasos futuros como variables predictoras (a menos que se especifique) y para que conservase la columna de fecha tras el filtrado.
4. Eso lo hice en un primer contacto con el framework hasta que funcionó correctamente y ahora estoy revisando uno a uno los ficheros de modelos para hacer tests en detalle (parámetros específicos de algunos métodos).
5. Por ahora lo he probado con los parámetros generales y funciona, genera un csv con las n caracteristicas seleccionadas, la variable objetivo y la fecha.
6. Aún no he probado los métodos de weka ni sklearn que vienen por separado, lo haré después de revisar sus códigos.

- Info:
1. Zotero.

- Feedback:
1. Algoritmo de predicción sencillo XGBoost o KNN (media de los n vecinos más cercanos).
2. Probar dataset completo y probar con mismo entorno de training y test.
3. Introducir predicción paramétrica (empezar con KNN y añadir otros únicamente cambiando parámetros).
4. Ver correlaciones entre métodos de selección y métodos de predicción.
5. Terminar primero con la selección de atributos y probar posteriormente con predicción.


CONCLUSIÓN: el objetivo del framework es dar una forma de comparar métodos de selección y predicción. Obviamente dependerá del tipo de dataset, propios parámetros de cada método, etc. Pero se puede probar en igualdad de condiciones.

## 11/10/2025 ##
- Análisis de automatic_selection.py: cambiado signo _evaluate_feature_set (está preparado para que menor valor absoluto, menor error) neg MSE mientras más alto (menor valor absoluto) devuelva, mejor.

- Terminado automatic_selection.py, queda model_based.py. Después probar time_series_specific.py, weka_inspired.py y sklearn_filter.py.

## 17/10/2025 ##
- Análisis de model_based hecho: Lasso penalización usando abs(coef regresion) L1 y Ridge (en Elastic Net) usando cuadrados. Lasso mejora disminuyendo la varianza (error por supersensibilidad) y Ridge bajando el bias (error por simplicidad).

- He empezado a hacer un notebook para probar con KNN sobre los csv generados por los distintos métodos de selección de atributos. He hecho el de PCA y el dataset A_data (del de los valores de stocks).

- Siguiente: Continuar con más métodos de selección.

## 18/10/2025 ##
- Hecha la comparación para todos los ficheros generados. 
- Pensar en: crear una celda para ejecutar el framework con todos los métodos (sólo he dejado fuera genéticos, dejar fuera también rfe). De esa forma poder cambiar los parámetros una vez y no tener que ejecutar el comando para cada método.
- Pensar también en probar otros métodos de predicción como XGBoost.

## 22/10/2025 ##
- REUNION: 
    * He analizado los códigos de los métodos de selección para entenderlos, aún sin weka ni sklearn.
    * He estado creando un notebook para comparar métodos sobre ficheros generados por el framework: Para un mismo csv original, he generado todos los csv's filtrados utilizando los distintos métodos de selección de atributos (aun sin weka y sklearn que tienen métodos aparte).
    * Mi idea es expandir el notebook para otros ficheros y/o cambiar parámetros aunque no estoy seguro de cómo hacerlo. Ahora mismo tengo que generar todos los tests en la carpeta de resultados y persiste el último test realizado.

    * Feedback:
        * Continuar con KNN. 
        * Hacerlo lo más amplio posible: tratar de utilizar también los métodos de weka y sklearn.
        * Revisar ventanado y escalado.
- Reflexión:
    * El objetivo final del framework es: dado un fichero csv de una serie temporal en principio con fv (variable objetivo) fija; comparar los distintos métodos de selección de atributos probando distintos parámetros fh (horizonte), ph (historial), n_features (número de variables seleccionadas), y su desempeño en la predicción con KNN.

## 29/10/2025 ##
- Revisando los últimos resultados Granger no es correcto para el flow actual. Granger trabaja con las variables sin lags, toma un par de características y comprueba en qué medida x causa a y: considera un modelo lineal hasta max_lag y comprueba si incluir x mejora la predicción usando solo y. Este método solo es útil para seleccionar características, no sus lags. **Probablemente descartado**

- Lasso selecciona una característica extraña y he acabado añadiendo para que tanto Lasso como Elastic Net admitan el parámetro alpha.

- Random Forest selecciona una con importancia significativa y el resto son despreciables. **REVISAR**

- Sequential todo 0, pero creo recordar que no era erróneo. **REVISAR**

- Después de revisar esos, empezar a ver weka_inspired (probablemente no estén implementados para seguir el mismo flujo actual).

## 4/11/2025 ##
- Random forest revisado, el resultado "extraño" es por las carácteristicas del dataset.

- Sequential revisado, las importancias realmente no son 0 pero son casi nulas.

- Analizado sklearn_filter methods y modificado el framework para soportar parámetros relacionados.