# Script de prueba para el paquete zumba_churn_pkg instalado.

import pandas as pd
from zumba_churn_pkg import predict
import warnings

# Suprimir advertencias de XGBoost o scikit-learn
warnings.filterwarnings("ignore")

# --- Configuración ---
# RUTA al archivo de datos de prueba
# ASEGÚRATE de que este archivo exista en el mismo directorio.
DATA_FILE = "raw_NewPredictChurnData.csv" 

def run_test():
    """Carga los datos de prueba y realiza una predicción usando el paquete instalado."""
    print("--- 1. Carga de datos de prueba ---")
    try:
        # Cargar los primeros 100 registros para una prueba rápida
        df_test = pd.read_csv(DATA_FILE).head(100)
        print(f"Éxito: Se cargaron {len(df_test)} filas del archivo {DATA_FILE}.")
    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo {DATA_FILE}. Por favor, verifique la ruta.")
        return

    print("--- 2. Realizando predicciones ---")
    try:
        # La función predict internamente carga el modelo y el binarizador
        results = predict(df_test)
        
        print("Éxito: Predicciones generadas correctamente.")
        print("\n--- Resultados de las primeras 5 predicciones ---")
        print(results.head())
        
        # Validación simple
        if not results.empty and "predicted_class" in results.columns:
            print("\nPrueba OK: La estructura de la salida es correcta.")
        else:
            print("\nPrueba FALLIDA: El DataFrame de resultados tiene una estructura incorrecta.")

    except Exception as e:
        print(f"ERROR al realizar la predicción: {e}")

if __name__ == "__main__":
    run_test()
