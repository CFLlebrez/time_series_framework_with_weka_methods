import pandas as pd
import numpy as np
# Modificar csv a evaluar
df = pd.read_csv(r"results\osuna_clean_output\TEST_WEKA_RELIEFF\transformed_data_fv4_fh3_ph1.csv")

# Ver rangos de todas las columnas numéricas
num_cols = df.select_dtypes(include='number').columns
stats = df[num_cols].agg(['min', 'max', 'std']).T
stats['range'] = stats['max'] - stats['min']
stats['n_unique'] = df[num_cols].nunique()
print(stats.sort_values('range', ascending=False).to_string())