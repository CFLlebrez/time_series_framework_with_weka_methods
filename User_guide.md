## Cómo usar el framework?
El comando base es<br> 
``python time_series_framework.py ruta_fichero_entrada nombre_fichero_salida --fv valor_fv --fh valor_fh --ph valor_ph --feature_selection --fs_method metodo_seleccion --fs_n_features valor_n_features --time_col nombre_columna_tiempo
``<br>

El fichero de entrada lo buscará en la carpeta ``input_csv_files`` y requiere la terminación ``.csv``. El nombre del fichero de salida se refiere el nombre base del cual se partirá para generar la carpeta en ``nombre_output`` dentro de la carpeta ``results``. Esa carpeta contendrá distintas carpetas y ficheros de resultados.

El parametro ``--time_col`` debe ser el nombre de la columna que indica el tiempo, ya sea fecha u hora (con otro tipo de columna de tiempo puede devolver resultados extraños).

# Métodos y parámetros propios:
Para seleccionar el método a utilizar debe estar el parámetro ``--feature_selection`` (se trata de un flag) y el parametro ``--fs_method`` que acepta valores 'pearson', 'ccf', 'mutual_info', 'random_forest', 'lasso', 'elastic_net', 'rfe', 'granger', 'pca', 'spectral', 'sequential', 'genetic', 'sklearn_filter', 'weka_inspired'.

Para los valores 'sklearn_filter' y 'weka_inspired' hay parámetros adicionales para seleccionar el método de sklearn o weka correspondiente:
* ``--sklearn_method`` con valores 'selectkbest', 'selectpercentile', 'genericunivariateselect', 'variancethreshold'
* ``--weka_inspired_method`` con valores 'cfs', 'relieff', 'infogain'

# Observaciones rápidas (osuna_clean fv=7, ph=5, fh=3, n=3):
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