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

## 12/11/2025 ##
- Métodos de weka son originales de Java, comprobar implementación y comparar con original. Mirar repositorios de github en caso de ser necesario.

## 25/11/2025
- Revisados los métodos de weka y añadidos al framework. Revisar algún resultado extraño (cfs solo selecciona 1, los resultados de relieff son un poco extraños) y añadir parámetros para esos métodos en el framework.

## 26/11/2025
- CFS: Es normal que tenga ese comportamiento por las características del dataset (las variables que más explican volume son sus propios lags pero a su vez presentan mucha redundancia, eso lleva a que sólo seleccione el lag más reciente). El número de características a seleccionar solo acota superiormente.
- InfoGain: En principio funcionaba con threshold, he cambiado el orden de los if para que priorice el numero de características a seleccionar.
- ReliefF: Lo único anómalo es que las características seleccionadas son lags anteriores cuando normalmente se selecciona el lag t-1.
- Añadir parámetros para los métodos.
- CFS: añadido número de retrocesos.
- InfoGain: añadido discretize y n_bins. Discretize es un flag, si se añade, True, si no, False.
- ReliefF: añadidos los parámetros, analizar su uso.

- Probando CFS con otros datasets siempre selecciona únicamente el lag t-1 de la variable objetivo. Probar con datasets con más características dado que estos al incluir la variable objetivo tienden a seleccionar sus propios lags.

- NEXT:
    - Probar con otros datasets.
    - Hacer el script para predicciones sobre los datasets filtrados.

## 15/12/2025
- Integrado Git LFS para los archivos .csv. (git lfs migrate info >> git lfs migrate import --include ...)
- Probado el dataset Osuna.csv. Mirar cómo hacer la limpieza de ese dataset para su utilización. Por ahora ";"->"," y "n/d"->"" manualmente.
- Problemas por lo pronto: columnas que son horas, y valores faltantes.

## 16/12/2025
- Notebook de limpieza de csv a modo de ejemplo. Para limpiar y testear el csv de Osuna.csv.
- Hecha la limpieza, he tratado de sustituir las horas faltantes por la media pero al no lograrlo he reemplazado por 12:00.
- Al probar la selección de atributos no da error, probar distintos métodos.
- Por lo pronto únicamente tiene importancia el propio lag de la variable objetivo.

## 29/12/2025
- Por hacer: 
    1. Probar algún csv en weka original y comparar con el implementado.
    2. Al usar csv's con datos tipo horas directamente eliminar.
    3. Revisar implementación de los que devuelven un solo atributo.

- Hecho:
    1. Limpiado correctamente el csv de osuna para hacer la prueba.
    2. Instalado Weka y probado CFS: Weka sí selecciona varios atributos mientras que la implementación únicamente uno: la variable objetivo t-1.

## 30/12/2025
- Los seleccionados el weka para predecir Se11VelVientoMax son Se11TMin_t-1, Se11VelViento_t-1, Se11VelVientoMax_t-1, Se11Precip_t-1
- He revisado la implementación y lo que ocurre es que Se11VelVientoMax_t-1 tiene mérito 1 como subset unitario.
- He empezado revisando la implementación del best_first_search que se basa en el cálculo de los méritos.
- Después he revisado el cálculo de los méritos pero está completamente acorde con el paper oficial.
- He probado a intentar forzar que seleccionase más características o que no pudiese seleccionar Se11VelVientoMax_t-1, pero los resultados siguen el mismo orden.
- Haciendo prints en puntos concretos se ven los distintos subsets y sus méritos y ninguno iguala la puntuación del set unitario.

-PROXIMO: probar otras implementaciones que haya o tratar de llamar a weka directamente.

-HECHO: probado el wrapper, da los mismos resultados que Weka (es llamar a weka desde python).

ejecutar .venv\Scripts\activate para el entorno virtual y seguir los comandos del notebook test_weka_wrapper.ipynb .

## 02/01/2026
- Creada una user guide, anotaciones en "observaciones rápidas"

## 11/01/2026
- Probados los métodos genéricos, decisiones tomadas (algunos quedan por revisar).

- Tras revisar, decisiones: 
    * Descartar: Granger, redundante por la lógica que sigue; Spectral y PCA, que evalúan características en general, no para una predicción específica; Pearson, por la implementación del framework es equivalente a CCF.

- Probar los de weka y sklearn y tomar decisiones.

## 14/01/2026
- Probados weka y sklearn. Probablemente Weka descartado por diferencias con el framework original y resultados incoherentes.

## 15/01/2026
- Reflexión: Hasta ahora he cambiado el flujo inicial al que pienso que tiene más sentido; traté de ejecutar los distintos métodos implementados comprobando su funcionamiento; hice una primera prueba de predicciones en base a los resultados obtenidos; añadí al framework los parámetros y las llamadas necesarias para incluir los métodos de sklearn y weka; instalé weka para comparar los resultados de la implementación y revisé los algoritmos para tratar de corregirlos sin éxito; probé a implementar métodos de weka llamando a un wrapper en caso de querer utilizarlo; incluí un nuevo csv más largo que traté de limpiar para que el framework pudiese ejecutarse con este fichero como entrada; volví a ejecutar todos los métodos sobre ese csv mientras hacía anotaciones en la guía de usuario y tomaba decisiones de qué métodos quedarían descartados.

Lo siguiente es confirmar que va por buen camino y concretar cómo se quiere el framework final, implementar las funcionalidades necesarias y finalizar después de realizar tests y comprobar su funcionamiento.

## 22/01/2026
- Próximos pasos después de comprobación de Pablo Reina:
    * Ver el tema de mantener la variable objetivo original.
    * Probar CFS con un dataset propio/kaggle
    * Probar a hacer predicciones sin selección (todos los atributos) y tras la selección de los distintos atributos para ver que da resultados con sentido en general.

- Hecho:
    * Mantener variable objetivo original: comprobado con ria1_test_transformer (contiente csv transformado de estacion.csv). Revisado el código de time_series_transformer_optimized.py y tiene sentido. El resultado consta de time_col, variables_t-ph, variables_t-ph+1, ..., fv, fv_t+1, ..., fv_t+fh.
    * Creado "own_dataset.csv" en el que y=sin(x1) + sin(x2).
    * Probado CFS en el dataset propio y vuelve a tener el mismo problema: selecciona y_t-1.

- Lo que queda es hacer pruebas de predicción antes y después de la selección.

- Re-adaptado el transformer para contener todos los lags incluyendo el instante actual de todas las variables y los instantes futuros de la variable objetivo.

- Siguiente, probar de nuevo que tras los cambios las selecciones funcionen correctamente y pasar a las predicciones.

## 26/01/2026
- Probadas algunas selecciones con own_dataset. Lasso da importancia 0 a todas las variables, probablemente por tener importancia baja.

- Terminar probando el resto y creando un script para predicción.

- # Observaciones rápidas (osuna_clean fv=7, ph=5, fh=3, n=3):
- Dada la implementación del framework ccf y pearson son equivalentes (pearson probaría lags). *Pearson descartable*
- Sequential necesita más trabajo (no funciona). *Revisar* || **Revisado**: sí funciona pero las importancias son del orden e-28.
- Genetic ha seleccionado 22 atributos (tarda mucho). *Descartable*
- Random forest devuelve un solo atributo (selecciona 3 pero solo uno con importancia no nula). *Revisar* || **Revisado**: similar a sequential pero orden e-08.
- Lasso devuelve generalmente menos atributos dado que penaliza los de baja importancia hasta bajarlos a 0.
- Elastic Net con l1_ratio=0.5 y alpha=1 (parametros default) parece que ha devuelto resultados con sentido.
- RFE tarda en ejecutar pero devuelve resultados con sentido.
- Granger está descartado dado que lo que hace es comparar lags de una característica (a una característica predictora le hace lags y los evalúa, pero en las candidatas ya se encuentran los lags. Esto hace que no tenga sentido utilizar este método en este framework). *Descartable*
- PCA devuelve características que no tienen mucho sentido (Devuelve DIA y lags de DIA cuando esa característica solo indica el día del año). *Descartable*
- Spectral más de lo mismo. *Descartable*. Estos tres métodos no tratan de analizar su importancia para predecir una variable en concreto sino su contribución en la serie en general.

- SelectKBest y SelectPercentile funcionan correctamente.
- GenericUnivariateSelect funciona, añadidos 'param' y 'strategy' aunque es complicado ajustar el número de características a seleccionar. Modos (strategy) 'fpr', 'fdr' y 'fwe' y param es el umbral de p-values máximo (se seleccionan las características cuyo p-value quede por debajo).

- CFS tiene el mismo problema: selecciona una sola característica. El lag t-1 de la variable objetivo tiene demasiada relevancia.
- Infogain parece que funciona correctamente pero en el framerwork original no puede usarse por las características del fichero.
- ReliefF devuelve características poco relevantes, distintas a las obtenidas en el framework original.


## 24/02/2026
- Evaluar el resto de lineales y no lineales con los datasets propios.
- Evaluados mutual_info, lasso y elastic_net,
rfe, selectkbest y percentile
pca y spectral
- Implementar los scripts para predicción y empezar comparando con y sin selección de atributos sobre varios datasets.


## 26/02/2026
- Reunión:
    * Evolutivos paralelizar: máquinas por VPN de la US.
    * CFS no evaluar la búsqueda sino la medida (probar con otra búsqueda que no sea tan rápida)
    * Implementar predicción y evaluar correctamente con benchmarks los métodos.
- Buscar csv's con características concretas: número de variables, tamaño dataset, datasets propios (Benchmarks - Datasets de Pablo Reina en github en el Readme).
- Realizar predicciones con los distintos métodos de selección y compararlos con la predicción total (comparar estabilidad (índice de Jaccard), precisión (MAE, RMSE, R2, sMAPE...) y cómputo (tiempo de ejecución)).

- Últimas pruebas sin lags ni steps, revisar que el transformer siga generando correctamente los csv transformados.

## 4/03/2026
- Ajustar csv's generados (transformado y filtrado).
- Crear el script de prediccion ya planeado.

## 5/03/2026
- Modificado para que la selección no genere un csv filtrado sino un json con metadatos.
- Creado el script de predicción y evaluación.
- Siguiente: Que los resultados se almacenen en un fichero bien presentado.

## 9/03/2026
- Mejorada la consistencia de los reports (weka_inspired y sklearn_filter era el mismo para los métodos de cada pack entonces se sobreescribían).
- Añadida función para generar un report completo (se añaden los distintos métodos como filas) para cada combinación de parámetros de entrada.

Siguiente: Buscar la forma de generar resultados de una ejecución, evaluar las opciones e implementar la más indicada.