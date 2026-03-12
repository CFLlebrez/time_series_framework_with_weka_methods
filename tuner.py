import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from evaluation import PredictiveEvaluator
import os
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Espacios de búsqueda
SEARCH_SPACES = {
    'knn': [3, 5, 7, 9, 11, 15, 20, 30, 35, 40, 50]
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


# Espacios de búsqueda representativos
FS_SEARCH_SPACES = {
    'lasso': {'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0]},
    'elastic_net': {'alpha': [0.001, 0.01, 0.1, 1.0], 'l1_ratio': [0.2, 0.5, 0.8]},
    'random_forest': {'max_depth': [5, 10, 20, None]},
    'mutual_info': {'n_neighbors': [3, 5, 7]}
}

def get_best_fs_params(method, transformed_csv, target_col, time_col, n_features, **kwargs):
    from feature_selection import select_features
    import os
    
    if method not in FS_SEARCH_SPACES:
        return kwargs

    print(f"[TUNING] Buscando parámetros óptimos para {method} (n_features={n_features})...")
    
    space = FS_SEARCH_SPACES[method]
    param_name = list(space.keys())[0] 
    values = space[param_name]
    
    best_value = values[0] # Inicializamos con el primero por seguridad (evita el None)
    best_score = float('inf')
    
    df = pd.read_csv(transformed_csv)
    y = df[target_col]
    
    # Limpiar kwargs para evitar duplicados en la llamada a select_features
    # Eliminamos verbose si ya existe para pasarlo nosotros
    test_params = kwargs.copy()
    test_params.pop('verbose', None)

    for val in values:
        test_params[param_name] = val
        # Forzamos n_features para que la comparativa sea justa
        test_params['n_features'] = n_features 
        
        try:
            # Directorio temporal para el tuning
            temp_dir = os.path.join("temp_tuning", method)
            os.makedirs(temp_dir, exist_ok=True)
            
            res = select_features(
                transformed_csv, temp_dir, target_col, 
                method=method, time_col=time_col, 
                generate_report=False, generate_filtered_csv=False,
                verbose=False, # Pasado explícitamente
                **test_params
            )
            
            selected_features = res['selected_features']
            importances = res.get('feature_importances', {})

            if len(selected_features) == 0: continue

            # FORZAR EL RECORTE A N_FEATURES:
            # Si Lasso devuelve 3 pero pedimos 2, ordenamos por importancia y cortamos.
            if len(selected_features) > n_features:
                # Ordenar las features seleccionadas según su importancia guardada en res
                sorted_feats = sorted(selected_features, 
                                    key=lambda x: abs(importances.get(x, 0)), 
                                    reverse=True)
                current_selected = sorted_feats[:n_features]
            else:
                current_selected = selected_features

            # Evaluamos el set recortado para ver si este alpha es realmente bueno
            X_subset = df[current_selected]
            score = quick_cv_eval(X_subset, y)
            
            print(f"  > Parámetro {param_name}={val}: RMSE Interno = {score:.4f}")
            
            if score < best_score:
                best_score = score
                best_value = val
                
        except Exception as e:
            # Si falla un valor, probamos el siguiente
            continue

    # Asegurarnos de que n_features vuelve en el diccionario
    kwargs['n_features'] = n_features 
    kwargs[param_name] = best_value
    
    print(f"[TUNING] Finalizado. Mejor {param_name} encontrado: {best_value} para n_features={n_features}")
    return kwargs

def quick_cv_eval(X, y):
    """Evaluación rápida con KNN y 3 splits para comparar sets de atributos."""
    tscv = TimeSeriesSplit(n_splits=3)
    errors = []
    
    for train_idx, test_idx in tscv.split(X):
        # Split
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Escalar
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)
        
        # Modelo rápido (K=5 por defecto para comparar)
        model = KNeighborsRegressor(n_neighbors=5)
        model.fit(X_tr_s, y_train)
        preds = model.predict(X_te_s)
        
        errors.append(np.sqrt(mean_squared_error(y_test, preds)))
        
    return np.mean(errors)