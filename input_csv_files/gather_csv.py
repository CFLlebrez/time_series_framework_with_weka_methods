import requests
import pandas as pd

# Parámetros de ejemplo (hay que sustituir con valores reales)
codigoProvincia = 41       # Sevilla
codigoEstacion = 2     # código real de estación, no "SE01"
fhInicio = "2022-01-01"
fhFin = "2022-01-15"
lgEt0 = "false"

url = f"https://www.juntadeandalucia.es/agriculturaypesca/ifapa/riaws/datosdiarios/{codigoProvincia}/{codigoEstacion}/{fhInicio}/{fhFin}/{lgEt0}"

headers = {"User-Agent": "Mozilla/5.0"}  # simula navegador

resp = requests.get(url, headers=headers)
resp.raise_for_status()

data = resp.json()
df = pd.DataFrame(data)
df.to_csv("input_csv_files/estacion.csv", index=False, encoding="utf-8")

print("✅ Datos guardados en estacion.csv")
