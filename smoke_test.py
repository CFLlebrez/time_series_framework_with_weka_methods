import pandas as pd
import numpy as np
import os
import subprocess
import json
import shutil

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
    # Obtener la ruta absoluta para evitar el error de carpetas duplicadas
    current_dir = os.getcwd()
    input_path = os.path.abspath(os.path.join(current_dir, "input_csv_files", f"{name}.csv"))
    output_base = os.path.abspath(os.path.join(current_dir, "results", "smoke_tests", f"{name}_output"))
    
    # Asegurarnos de que el archivo existe antes de lanzar el comando
    if not os.path.exists(input_path):
        print(f"❌ ERROR: El archivo de entrada no existe en {input_path}")
        return

    print(f"\n" + "="*60)
    print(f">>> SMOKE TEST: {name.upper()} | MÉTODO: {method}")
    print(f"Ruta: {input_path}")
    print("="*60)

    cmd = [
        "python", "time_series_framework.py",
        input_path,             
        output_base,            
        "--fv", str(fv),
        "--fh", "0",
        "--ph", "1",
        "--time_col", "fecha",
        "--feature_selection",
        "--run_name", f"smoke_test_{method}",
        "--fs_method", method,
        "--fs_n_features", str(n_expected),
        "--evaluation"           
    ]
    
    try:
        # Usamos shell=True solo si estás en Windows y tienes problemas con las rutas, 
        # pero con abspath no debería ser necesario.
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode != 0:
            print(f"❌ ERROR en el framework:\n{process.stderr}")
            # Imprimimos también stdout por si el error está allí
            print(f"STDOUT:\n{process.stdout}")
            return
        
        # Validar el JSON de salida
        # Nota: La carpeta de salida según tu framework parece ser feature_selection_{method}
        json_path = f"results/smoke_tests/{name}_output/smoke_test_{method}/feature_selection_{method}/selection_metadata_{method}.json"

        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                meta = json.load(f)
            
            selected = meta.get('selected_features', [])
            n_actual = len(selected)
            
            if n_actual == n_expected:
                print(f"✅ ÉXITO: Seleccionados {n_actual} atributos: {selected}")
            else:
                print(f"❌ FALLO: Se esperaban {n_expected} atributos, se obtuvieron {n_actual}.")
                print(f"Atributos obtenidos: {selected}")
        else:
            print(f"❌ ERROR: No se encontró el JSON de metadatos en: {json_path}")
            
    except Exception as e:
        print(f"💥 ERROR DE EJECUCIÓN: {e}")

if __name__ == "__main__":        
    generate_smoke_data()
    
    # Índices FV:
    # test_lineal -> target está en el índice 5
    # test_sinusoidal -> target está en el índice 3
    
    run_test("test_lineal", "lasso", 2, 5) 
    run_test("test_sinusoidal", "random_forest", 1, 3)