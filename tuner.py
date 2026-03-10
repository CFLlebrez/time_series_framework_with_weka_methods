import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from evaluation import PredictiveEvaluator

# Espacios de búsqueda
SEARCH_SPACES = {
    'knn': [3, 5, 7, 9, 11, 15],
    'lasso': [0.001, 0.01, 0.1, 1.0],
    'elastic_net': [0.001, 0.01, 0.1, 1.0],
    'random_forest': [5, 10, 20, None], # max_depth
    'mutual_info': [3, 5, 7] # n_neighbors
}

def tune_knn_k(df, feature_cols, target_cols):
    """Búsqueda del mejor K para KNN sobre el set de entrenamiento."""
    print(f"[TUNING] Optimizando K para KNN (Espacio: {SEARCH_SPACES['knn']})...")
    best_k = 5
    best_rmse = float('inf')
    
    # Usamos un split más pequeño para no acomplejar el tuning
    evaluator = PredictiveEvaluator(n_splits=3)
    
    for k in SEARCH_SPACES['knn']:
        evaluator.n_neighbors = k
        results = evaluator._evaluate_model(df, feature_cols, target_cols)
        if results['rmse_avg'] < best_rmse:
            best_rmse = results['rmse_avg']
            best_k = k
    return best_k

def get_best_fs_params(method, X, y, target_col):
    """Encuentra parámetros óptimos para el selector de atributos."""
    # Nota: Aquí se implementaría un bucle similar al de KNN si quisiéramos
    # que el selector también se 'auto-ajustara'. Por ahora, si el usuario
    # no define parámetros, este es el lugar para asignar valores por defecto inteligentes.
    pass