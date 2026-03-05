import pandas as pd
import numpy as np
import json
import os
import time
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class PredictiveEvaluator:
    def __init__(self, n_splits=5, n_neighbors=5):
        """
        Inicializa el evaluador con validación cruzada temporal.
        
        Args:
            n_splits (int): Número de particiones para TimeSeriesSplit.
            n_neighbors (int): K para el algoritmo KNN.
        """
        self.n_splits = n_splits
        self.n_neighbors = n_neighbors
        self.tscv = TimeSeriesSplit(n_splits=n_splits)

    def run_full_evaluation(self, csv_transformed_path, json_metadata_path, time_col):
        """
        Carga los metadatos y el CSV para ejecutar la evaluación comparativa entre
        el set completo de variables (Baseline) y el set filtrado.
        """
        # 1. Cargar metadatos del JSON generado en la fase de selección
        if not os.path.exists(json_metadata_path):
            raise FileNotFoundError(f"No se encontró el archivo de metadatos: {json_metadata_path}")
            
        with open(json_metadata_path, 'r') as f:
            meta = json.load(f)
        
        selected_features = meta['selected_features']
        target_columns = meta['target_columns']
        method_name = meta['experiment_info']['method']
        
        # 2. Cargar Dataset Transformado (Fuente de verdad)
        df = pd.read_csv(csv_transformed_path)
        
        # 3. Identificar todas las variables predictoras para el Baseline
        # Excluimos targets, posibles columnas de tiempo e índices
        all_features = [c for c in df.columns if c not in target_columns 
                        and c.lower() != time_col]
        
        print(f"\n" + "="*60)
        print(f"SISTEMA DE EVALUACIÓN DE PREDICCIÓN (Horizonte FH: {len(target_columns)})")
        print("="*60)
        
        # 4. Evaluar Baseline (Todas las variables originales + lags)
        print(f"\n[PASO 1/2] Evaluando Baseline...")
        print(f"Variables utilizadas: {len(all_features)}")
        baseline_results = self._evaluate_model(df, all_features, target_columns)
        
        # 5. Evaluar Método Seleccionado
        print(f"\n[PASO 2/2] Evaluando Selección: {method_name}...")
        print(f"Variables utilizadas: {len(selected_features)}")
        method_results = self._evaluate_model(df, selected_features, target_columns)
        
        # 6. Generar reporte comparativo final
        return self._generate_final_report(baseline_results, method_results, method_name, 
                                           len(all_features), len(selected_features))

    def _evaluate_model(self, df, feature_cols, target_cols):
        """
        Ejecuta el core de la evaluación con TimeSeriesSplit.
        """
        X = df[feature_cols]
        y = df[target_cols]
        
        metrics = {'rmse': [], 'mae': [], 'r2': []}
        inference_times = []
        
        # Bucle de Validación Cruzada Temporal (Walk-Forward Validation)
        for train_index, test_index in self.tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # ESCALADO PROFESIONAL: Ajustar solo en entrenamiento para evitar fugas (Data Leakage)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Configuración del modelo KNN (Multi-output nativo)
            model = KNeighborsRegressor(n_neighbors=self.n_neighbors)
            model.fit(X_train_scaled, y_train)
            
            # MEDICIÓN DE TIEMPO DE INFERENCIA (Solo la fase de predicción)
            start_time = time.perf_counter()
            preds = model.predict(X_test_scaled)
            end_time = time.perf_counter()
            
            # Tiempo por fold ajustado al tamaño del set de test
            inference_times.append((end_time - start_time) / len(X_test))
            
            # Cálculo de métricas
            metrics['rmse'].append(np.sqrt(mean_squared_error(y_test, preds)))
            metrics['mae'].append(mean_absolute_error(y_test, preds))
            metrics['r2'].append(r2_score(y_test, preds, multioutput='uniform_average'))
            
        return {
            'rmse_avg': np.mean(metrics['rmse']),
            'mae_avg': np.mean(metrics['mae']),
            'r2_avg': np.mean(metrics['r2']),
            'time_avg': np.mean(inference_times)
        }

    def _generate_final_report(self, base, meth, method_name, n_base, n_meth):
        """
        Crea un DataFrame comparativo y calcula ganancias de eficiencia.
        """
        data = {
            'Métrica': ['Nº Variables', 'RMSE (Media)', 'MAE (Media)', 'R2 (Media)', 'T. Inferencia (s/sample)'],
            'Baseline (Todo)': [n_base, base['rmse_avg'], base['mae_avg'], base['r2_avg'], base['time_avg']],
            f'Filtrado ({method_name})': [n_meth, meth['rmse_avg'], meth['mae_avg'], meth['r2_avg'], meth['time_avg']]
        }
        
        report_df = pd.DataFrame(data)
        
        # Cálculos de resultados
        error_reduction = ((base['rmse_avg'] - meth['rmse_avg']) / base['rmse_avg']) * 100
        dim_reduction = (1 - (n_meth / n_base)) * 100
        time_improvement = ((base['time_avg'] - meth['time_avg']) / base['time_avg']) * 100
        
        print("\n" + "-"*60)
        print("RESUMEN DE RESULTADOS")
        print("-"*60)
        print(f"Reducción de Dimensionalidad: {dim_reduction:.2f}%")
        print(f"Mejora en Error (RMSE): {error_reduction:.2f}%")
        print(f"Mejora en Tiempo de Respuesta: {time_improvement:.2f}%")
        print("-"*60)
        
        return report_df