# Script de prueba para el paquete zumba_churn_pkg instalado.

import pandas as pd
from zumba_churn_pkg import predict 
import pytest
import warnings

# Suprimir advertencias de XGBoost o scikit-learn
warnings.filterwarnings("ignore")

# --- Configuración ---
# RUTA al archivo de datos de prueba
DATA_FILE = "data/raw_NewPredictChurnData.csv" 

@pytest.mark.integration
def test_package_predict_execution():
    """
    Prueba que el paquete se puede importar, cargar los datos de entrada
    y ejecutar la predicción correctamente, devolviendo la estructura esperada.
    """
    print(f"\nIntentando cargar datos de: {DATA_FILE}")
    try:
        # Cargar un subconjunto de datos para la prueba
        df_test = pd.read_csv(DATA_FILE).head(10)
        print(f"Éxito: Se cargaron {len(df_test)} filas.")
    except FileNotFoundError:
        # Esta prueba DEBE fallar si el archivo de datos no existe en la ruta esperada.
        pytest.skip(f"No se encontró el archivo de datos requerido: {DATA_FILE}")
        return
        
    # Realizar la predicción
    # USAR ARGUMENTO DE PALABRA CLAVE: input_data=
    results = predict(input_data=df_test)
    
    # 1. Asegurar que no haya errores
    assert results["errors"] is None, f"La predicción falló con el error: {results['errors']}"
    
    # 2. Asegurar que haya predicciones
    assert results["prediction"] is not None
    assert results["probability"] is not None
    
    # 3. Asegurar que el número de resultados coincide con el número de entradas
    assert len(results["prediction"]) == len(df_test)
    assert len(results["probability"]) == len(df_test)
    
    print("\nPrueba OK: Predicciones generadas correctamente con la estructura esperada.")
