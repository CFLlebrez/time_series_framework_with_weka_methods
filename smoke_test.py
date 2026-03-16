import pandas as pd
import numpy as np
import os
import subprocess
import json

def generate_smoke_data():
    """Genera datasets sintéticos controlados."""
    n_samples = 500
    t = np.linspace(0, 100, n_samples)
    os.makedirs('input_csv_files', exist_ok=True)
    
    # Dataset 1: Lineal (y depende de x1 y x2)
    # Índices: 0:fecha, 1:x1, 2:x2, 3:x3_ruido, 4:x4_ruido, 5:target
    df_lin = pd.DataFrame({
        'fecha': pd.date_range('2023-01-01', periods=n_samples, freq='h'),
        'x1': np.random.randn(n_samples),
        'x2': np.random.randn(n_samples),
        'x3_ruido': np.random.uniform(0, 100, n_samples),
        'x4_ruido': np.random.normal(0, 50, n_samples),
        'target': 0.0
    })
    df_lin['target'] = 2.5 * df_lin['x1'] - 1.2 * df_lin['x2'] + np.random.normal(0, 0.05, n_samples)
    df_lin.to_csv('input_csv_files/test_lineal.csv', index=False)

    # Dataset 2: No Lineal (Seno)
    # Índices: 0:fecha, 1:x_val, 2:x_random, 3:target
    df_nonlin = pd.DataFrame({
        'fecha': pd.date_range('2023-01-01', periods=n_samples, freq='h'),
        'x_val': np.sin(t),
        'x_random': np.random.randn(n_samples),
        'target': np.sin(t) + np.random.normal(0, 0.02, n_samples)
    })
    df_nonlin.to_csv('input_csv_files/test_sinusoidal.csv', index=False)

def run_test(name, method, n_expected, fv):
    # 1. Rutas absolutas
    current_dir = os.getcwd()
    input_path = os.path.abspath(os.path.join(current_dir, "input_csv_files", f"{name}.csv"))
    
    # Directorio base donde el framework creará la jerarquía
    output_base = os.path.abspath(os.path.join(current_dir, "results", "smoke_tests"))
    run_id = f"smoke_test_{method}"

    if not os.path.exists(input_path):
        print(f"❌ ERROR: El archivo de entrada no existe en {input_path}")
        return False, []

    print(f"\n" + "="*60)
    print(f">>> EJECUTANDO: {name.upper()} | MÉTODO: {method}")
    print("="*60)

    # 2. Comando para el framework
    cmd = [
        "python", "time_series_framework.py",
        input_path,             
        output_base,            
        "--fv", str(fv),
        "--fh", "0",
        "--ph", "1",
        "--time_col", "fecha",
        "--feature_selection",
        "--run_name", run_id,
        "--fs_method", method,
        "--fs_n_features", str(n_expected),
        "--evaluation"           
    ]
    
    try:
        # Ejecución
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode != 0:
            print(f"❌ ERROR en el framework:\n{process.stderr}")
            return False, []
        
        # 3. CONSTRUCCIÓN DE RUTA DE VALIDACIÓN (Ajustada a tu estructura)
        # direccion_base_repositorio/results/smoke_tests/test_{tipo_test}_output/smoke_test_{metodo}
        json_path = os.path.join(
            output_base, 
            f"{name}_output", 
            run_id, 
            f"feature_selection_{method}", 
            f"selection_metadata_{method}.json"
        )

        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                meta = json.load(f)
            
            selected = meta.get('selected_features', [])
            n_actual = len(selected)
            success = (n_actual == n_expected)
            
            if success:
                print(f"✅ ÉXITO: Seleccionados {n_actual} atributos: {selected}")
            else:
                print(f"❌ FALLO: Se esperaban {n_expected}, se obtuvieron {n_actual}.")
            
            return success, selected
        else:
            print(f"❌ ERROR: No se encontró el JSON en: {json_path}")
            return False, []
            
    except Exception as e:
        print(f"💥 ERROR DE EJECUCIÓN: {e}")
        return False, []

def run_full_battery():
    os.makedirs('input_csv_files', exist_ok=True)
    generate_smoke_data() # Asegúrate de que esta función esté definida arriba

    # Configuración de tests
    tests = [
        # --- LINEALES (Dataset: test_lineal) ---
        ("test_lineal", 5, "lasso", 2),
        ("test_lineal", 5, "elastic_net", 2),
        ("test_lineal", 5, "pearson", 2), # Correlación simple
        ("test_lineal", 5, "ccf", 2),     # Correlación cruzada
        
        # Selectores de Sklearn (Filtros)
        ("test_lineal", 5, "sklearn_filter", 2), # Probará SelectKBest por defecto
        
        # Wrappers (Lineales)
        ("test_lineal", 5, "sequential", 2),     # Forward/Backward selection
        
        # --- NO LINEALES (Dataset: test_sinusoidal) ---
        ("test_sinusoidal", 3, "random_forest", 1),
        ("test_sinusoidal", 3, "mutual_info", 1),
        ("test_sinusoidal", 3, "rfe", 1),          # Recursive Feature Elimination
    ]

    results_summary = []
    for d_name, fv, method, n_exp in tests:
        # Ahora run_test devuelve (success, selected)
        success, selected = run_test(d_name, method, n_exp, fv)
        
        results_summary.append({
            "Dataset": d_name,
            "Metodo": method,
            "Status": "✅ OK" if success else "❌ FAIL",
            "Variables": ", ".join(selected) if selected else "Ninguna"
        })
    
    # Mostrar resumen al final
    print("\n" + "="*70)
    print("RESUMEN DE BATERÍA DE TESTS")
    print("="*70)
    df_res = pd.DataFrame(results_summary)
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    run_full_battery()