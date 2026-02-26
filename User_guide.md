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

Métodos: 
- Lineales: Lasso, Elastic Net, SelectKBest, SelectPercentile, CCF, Sequential (usa linear regression)
- No Lineales: Random Forest, MI, RFE
- Generales: Spectral, PCA
- Mal funcionamiento: Genetic (tiempo y resultados no siempre congruentes), ReliefF (mal resultado)
- No aportan: Pearson (cfs), Granger (incompatible), CFS (únicamente una variable, distinto al original), InfoGain (incompatible)
