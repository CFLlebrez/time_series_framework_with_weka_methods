# Arquitectura del Módulo de Selección de Atributos

## Estructura General

```
feature_selection/
├── __init__.py                  # Exporta las clases y funciones principales
├── base.py                      # Clase base y utilidades comunes
├── correlation_based.py         # Métodos basados en correlación
├── model_based.py               # Métodos basados en modelos
├── time_series_specific.py      # Métodos específicos para series temporales
├── automatic_selection.py       # Métodos de selección automática
├── visualization.py             # Funciones para visualizar resultados
└── utils.py                     # Funciones de utilidad
```

## Diseño de Clases

### Clase Base

```python
class FeatureSelector:
    """Clase base para todos los métodos de selección de atributos."""
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.feature_importances_ = None
        self.selected_features_ = None
        
    def fit(self, X, y=None):
        """Ajusta el selector a los datos."""
        raise NotImplementedError("Las subclases deben implementar este método")
        
    def transform(self, X):
        """Transforma los datos usando solo las características seleccionadas."""
        if self.selected_features_ is None:
            raise ValueError("El selector debe ser ajustado antes de transformar")
        return X[self.selected_features_]
        
    def fit_transform(self, X, y=None):
        """Ajusta el selector y transforma los datos."""
        self.fit(X, y)
        return self.transform(X)
        
    def get_feature_importances(self):
        """Devuelve las importancias de las características."""
        if self.feature_importances_ is None:
            raise ValueError("El selector debe ser ajustado antes de obtener importancias")
        return self.feature_importances_
        
    def get_selected_features(self):
        """Devuelve las características seleccionadas."""
        if self.selected_features_ is None:
            raise ValueError("El selector debe ser ajustado antes de obtener características")
        return self.selected_features_
```

### Interfaz de Línea de Comandos

```python
def main():
    parser = argparse.ArgumentParser(description='Selección de atributos para series temporales.')
    parser.add_argument('input_file', type=str, help='Archivo CSV de entrada')
    parser.add_argument('output_file', type=str, help='Archivo CSV de salida con atributos seleccionados')
    parser.add_argument('--method', type=str, required=True, 
                        choices=['pearson', 'ccf', 'mutual_info', 'random_forest', 
                                'lasso', 'elastic_net', 'rfe', 'granger', 'pca', 
                                'spectral', 'sfs', 'sbs', 'genetic'],
                        help='Método de selección de atributos')
    parser.add_argument('--target', type=str, required=True, 
                        help='Columna objetivo para la predicción')
    parser.add_argument('--n_features', type=int, default=None,
                        help='Número de características a seleccionar')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Umbral para la selección de características')
    parser.add_argument('--max_lag', type=int, default=10,
                        help='Máximo lag a considerar para métodos basados en series temporales')
    parser.add_argument('--visualize', action='store_true',
                        help='Generar visualizaciones de importancia de características')
    
    args = parser.parse_args()
    
    # Cargar datos
    df = pd.read_csv(args.input_file)
    
    # Crear selector según el método elegido
    selector = create_selector(args.method, args.n_features, args.threshold, args.max_lag)
    
    # Preparar datos para selección de características
    X, y = prepare_data_for_feature_selection(df, args.target, args.max_lag)
    
    # Ajustar selector
    selector.fit(X, y)
    
    # Obtener características seleccionadas
    selected_features = selector.get_selected_features()
    
    # Transformar datos originales
    df_selected = selector.transform(df)
    
    # Guardar resultados
    df_selected.to_csv(args.output_file, index=False)
    
    # Generar visualizaciones si se solicita
    if args.visualize:
        visualize_feature_importances(selector, args.output_file.replace('.csv', '_importances.png'))
    
    print(f"Selección de características completada. Se seleccionaron {len(selected_features)} características.")
    print(f"Resultados guardados en {args.output_file}")
```

## Integración con el Framework Existente

El módulo de selección de atributos se integrará con el framework existente de dos maneras:

1. **Como preprocesamiento**: Antes de la transformación de series temporales, para seleccionar qué variables y lags usar.

2. **Como postprocesamiento**: Después de la transformación, para seleccionar las características más relevantes del conjunto de datos transformado.

### Modificación del script principal

```python
# Añadir argumentos para selección de características
parser.add_argument('--feature_selection', action='store_true',
                    help='Aplicar selección de características')
parser.add_argument('--fs_method', type=str, default='pearson',
                    choices=['pearson', 'ccf', 'mutual_info', 'random_forest', 
                            'lasso', 'elastic_net', 'rfe', 'granger', 'pca', 
                            'spectral', 'sfs', 'sbs', 'genetic'],
                    help='Método de selección de características')
parser.add_argument('--fs_n_features', type=int, default=None,
                    help='Número de características a seleccionar')
parser.add_argument('--fs_threshold', type=float, default=None,
                    help='Umbral para la selección de características')
parser.add_argument('--fs_max_lag', type=int, default=None,
                    help='Máximo lag a considerar (por defecto, igual a ph)')
```

## Flujo de Trabajo

1. El usuario proporciona un archivo CSV con series temporales multivariables.
2. Opcionalmente, el usuario especifica un método de selección de atributos y sus parámetros.
3. Si se solicita selección de atributos, el sistema:
   a. Prepara los datos para la selección de atributos (creando lags si es necesario).
   b. Aplica el método de selección especificado.
   c. Genera visualizaciones de importancia de atributos si se solicita.
4. El sistema transforma los datos usando solo las variables y lags seleccionados.
5. El resultado se guarda en un archivo CSV listo para regresión.

## Consideraciones de Rendimiento

- Para conjuntos de datos grandes, implementar procesamiento por lotes.
- Paralelizar cálculos cuando sea posible (especialmente para métodos computacionalmente intensivos).
- Cachear resultados intermedios para evitar recálculos.
- Proporcionar indicadores de progreso para métodos de larga duración.
