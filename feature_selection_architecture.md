# Arquitectura del Módulo de Selección de Atributos

## Estructura General

```
feature_selection/
├── __init__.py                  # Exporta las clases y funciones principales
├── automatic_selection.py       # Métodos de selección automática
├── correlation_based.py         # Métodos basados en correlación
├── model_based.py               # Métodos basados en modelos
├── sklearn_filter.py            # Métodos de filtro de sklearn
├── time_series_specific.py      # Métodos específicos para series temporales
├── visualization.py             # Funciones para visualizar resultados
└── weka_inspired.py             # Métodos basados en Weka
```
No existen base.py, utils.py

## Diseño de Clases

### Clase Base

```python
class BaseFeatureSelector:
    """Clase base para todos los métodos de selección de atributos."""
    
    def __init__(self, n_features=None, threshold=None, verbose=False):
        """
        Inicializa el selector de características.
        
        Args:
            n_features (int, optional): Número de características a seleccionar.
            threshold (float, optional): Umbral para la selección de características.
            verbose (bool, optional): Si es True, muestra información detallada.
        """
        self.n_features = n_features
        self.threshold = threshold
        self.verbose = verbose
        self.feature_importances_ = None
        self.selected_features_ = None
        
    def fit(self, X, y=None):
        """
        Ajusta el selector a los datos.
        
        Args:
            X (DataFrame): Datos de entrada.
            y (Series, optional): Variable objetivo.
            
        Returns:
            self: El selector ajustado.
        """
        raise NotImplementedError("Las subclases deben implementar este método")
        
    def transform(self, X):
        """
        Transforma los datos usando solo las características seleccionadas.
        
        Args:
            X (DataFrame): Datos de entrada.
            
        Returns:
            DataFrame: Datos transformados con solo las características seleccionadas.
        """
        if self.selected_features_ is None:
            raise ValueError("El selector debe ser ajustado antes de transformar")
        return X[self.selected_features_]
        
    def fit_transform(self, X, y=None):
        """
        Ajusta el selector y transforma los datos.
        
        Args:
            X (DataFrame): Datos de entrada.
            y (Series, optional): Variable objetivo.
            
        Returns:
            DataFrame: Datos transformados con solo las características seleccionadas.
        """
        self.fit(X, y)
        return self.transform(X)
        
    def get_feature_importances(self):
        """
        Devuelve las importancias de las características.
        
        Returns:
            Series: Importancias de las características.
        """
        if self.feature_importances_ is None:
            raise ValueError("El selector debe ser ajustado antes de obtener importancias")
        return self.feature_importances_
        
    def get_selected_features(self):
        """
        Devuelve las características seleccionadas.
        
        Returns:
            list: Nombres de las características seleccionadas.
        """
        if self.selected_features_ is None:
            raise ValueError("El selector debe ser ajustado antes de obtener características")
        return self.selected_features_
    
    def _select_features(self):
        """
        Selecciona las características basadas en sus importancias.
        
        Returns:
            list: Nombres de las características seleccionadas.
        """
        if self.feature_importances_ is None:
            raise ValueError("Las importancias de características deben calcularse antes de seleccionar")
        
        # Ordenar características por importancia
        sorted_features = self.feature_importances_.sort_values(ascending=False)
        
        # Seleccionar características basadas en n_features o threshold
        if self.n_features is not None:
            selected = sorted_features.iloc[:self.n_features].index.tolist()
        elif self.threshold is not None:
            selected = sorted_features[sorted_features >= self.threshold].index.tolist()
        else:
            # Por defecto, seleccionar características con importancia positiva
            selected = sorted_features[sorted_features > 0].index.tolist()
        
        return selected
    
    def plot_feature_importances(self, top_n=None, figsize=(10, 8), save_path=None):
        """
        Visualiza las importancias de las características.
        
        Args:
            top_n (int, optional): Número de características principales a mostrar.
            figsize (tuple, optional): Tamaño de la figura.
            save_path (str, optional): Ruta para guardar la figura.
            
        Returns:
            matplotlib.figure.Figure: La figura generada.
        """
        if self.feature_importances_ is None:
            raise ValueError("El selector debe ser ajustado antes de visualizar importancias")
        
        # Ordenar características por importancia
        sorted_importances = self.feature_importances_.sort_values(ascending=False)
        
        # Limitar a top_n si se especifica
        if top_n is not None:
            sorted_importances = sorted_importances.iloc[:top_n]
        
        # Crear figura
        fig, ax = plt.subplots(figsize=figsize)
        
        # Crear gráfico de barras
        sorted_importances.plot(kind='bar', ax=ax)
        
        # Añadir títulos y etiquetas
        ax.set_title('Importancia de las Características')
        ax.set_xlabel('Características')
        ax.set_ylabel('Importancia')
        
        # Rotar etiquetas del eje x para mejor legibilidad
        plt.xticks(rotation=90)
        
        # Ajustar diseño
        plt.tight_layout()
        
        # Guardar figura si se especifica ruta
        if save_path:
            plt.savefig(save_path)
        
        return fig
```

### Interfaz de Línea de Comandos

```python
def main():
    # Parsear argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Framework para series temporales con selección de atributos.')
    
    # Argumentos para el archivo de entrada/salida
    parser.add_argument('input_file', type=str, help='Ruta al archivo CSV de entrada')
    parser.add_argument('output_dir', type=str, help='Directorio para guardar los resultados')
    
    # Argumentos para la transformación de series temporales
    parser.add_argument('--fv', type=int, required=True, help='Forecast Variable - índice de la columna a predecir')
    parser.add_argument('--fh', type=int, required=True, help='Forecast Horizon - número de valores futuros a predecir')
    parser.add_argument('--ph', type=int, required=True, help='Past History - número de valores pasados a utilizar')
    
    # Argumentos para la selección de atributos
    parser.add_argument('--feature_selection', action='store_true', help='Aplicar selección de atributos')
    parser.add_argument('--fs_method', type=str, default='random_forest',
                        choices=['pearson', 'ccf', 'mutual_info', 'random_forest', 
                                'lasso', 'elastic_net', 'rfe', 'granger', 'pca', 
                                'spectral', 'sequential', 'genetic'],
                        help='Método de selección de atributos')
    parser.add_argument('--fs_n_features', type=int, default=None,
                        help='Número de atributos a seleccionar')
    parser.add_argument('--fs_threshold', type=float, default=None,
                        help='Umbral para la selección de atributos')
    parser.add_argument('--fs_max_lag', type=int, default=None,
                        help='Máximo lag a considerar (por defecto, igual a ph)')
    
    args = parser.parse_args()
    input_folder_dir = "input_csv_files/"+args.input_file
    output_folder_dir = "output_csv_files/"+args.output_dir
    # Crear directorio de salida si no existe
    os.makedirs(output_folder_dir, exist_ok=True)
    
    # Cargar datos
    print(f"Cargando datos desde {input_folder_dir}...")
    df = pd.read_csv(input_folder_dir)
    
    # Obtener nombre de la columna objetivo
    columns = df.columns.tolist()
    if args.fv < 0 or args.fv >= len(columns):
        raise ValueError(f"FV index {args.fv} está fuera de rango. Rango válido: 0-{len(columns)-1}")
    target_col = columns[args.fv]
    print(f"Variable objetivo: {target_col} (índice {args.fv})")
    
    # Determinar máximo lag para selección de atributos
    max_lag = args.fs_max_lag if args.fs_max_lag is not None else args.ph
    
    # Aplicar selección de atributos si se solicita
    if args.feature_selection:
        print(f"\nAplicando selección de atributos usando método '{args.fs_method}'...")
        
        # Ejecutar selección de atributos
        fs_results = select_features(
            input_folder_dir,
            os.path.join(output_folder_dir, 'feature_selection'),
            target_col,
            method=args.fs_method,
            n_features=args.fs_n_features,
            threshold=args.fs_threshold,
            max_lag=max_lag
        )
        
        # Mostrar características seleccionadas
        selected_features = fs_results['selected_features']
        print(f"\nCaracterísticas seleccionadas ({len(selected_features)}):")
        for i, feature in enumerate(selected_features, 1):
            importance = fs_results['feature_importances'].get(feature, 'N/A')
            if isinstance(importance, (int, float)):
                print(f"{i}. {feature} (importancia: {importance:.4f})")
            else:
                print(f"{i}. {feature} (importancia: {importance})")
        
        # Obtener ruta al CSV filtrado
        filtered_csv = fs_results['filtered_csv_path']
        print(f"\nCSV filtrado generado: {filtered_csv}")
        
        # Usar el CSV filtrado para la transformación
        input_for_transform = filtered_csv
    else:
        # Usar el CSV original para la transformación
        input_for_transform = input_folder_dir
    
    # Ejecutar transformación de series temporales
    print("\nEjecutando transformación de series temporales...")
    
    # Importar el script de transformación
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from time_series_transformer_optimized import transform_time_series
    
    # Definir archivo de salida para la transformación
    output_file = os.path.join(output_folder_dir, 'transformed_data.csv')
    
    # Ejecutar transformación
    transform_time_series(input_for_transform, output_file, args.fv, args.fh, args.ph)
    
    print(f"\nTransformación completada. Resultados guardados en: {output_file}")
    print("\nProceso completo finalizado.")
```

## Integración con el Framework Existente

El módulo de selección de atributos se integrará con el framework existente de dos maneras:

1. **Como preprocesamiento**: Antes de la transformación de series temporales, para seleccionar qué variables y lags usar.

2. **Como postprocesamiento**: Después de la transformación, para seleccionar las características más relevantes del conjunto de datos transformado.

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
