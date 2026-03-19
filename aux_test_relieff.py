import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv(r"results\osuna_clean_output\TEST_WEKA_RELIEFF\transformed_data_fv4_fh3_ph1.csv")

target_col = 'Se11TMed'
drop_cols = [c for c in df.columns if c.startswith(f"{target_col}_t+") or c == target_col or c == 'FECHA']
X = df.drop(columns=drop_cols).select_dtypes(include='number')
y = df[target_col]

# Ver correlación de Pearson de cada feature con el target (referencia)
corr = X.corrwith(y).abs().sort_values(ascending=False)
print("=== Correlación Pearson con target ===")
print(corr.to_string())

# Ver qué features dominan el espacio MinMax (varianza tras escalar)
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
print("\n=== Varianza tras MinMax (mayor = más dispersa en espacio normalizado) ===")
print(X_scaled.var().sort_values(ascending=False).to_string())