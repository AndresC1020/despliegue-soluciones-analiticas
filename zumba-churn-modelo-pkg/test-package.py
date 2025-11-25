import pandas as pd
import os
import warnings
from zumba_churn_pkg import predict 

# --- Configuración ---
# RUTA al archivo de datos de prueba (asumimos que está en el mismo directorio)
TEST_DATA_FILE = 'raw_NewPredictChurnData.csv'
OUTPUT_FILE_NAME = 'predictions_output.csv'

# Suprimir advertencias si aparecen
warnings.filterwarnings("ignore")

def run_prediction_test():
    """
    Carga los datos de prueba, ejecuta la predicción con el paquete 
    instalado, imprime los resultados en la consola y los guarda en un nuevo CSV.
    """
    print("=========================================================")
    print(f"--- Prueba de Predicción para Paquete '{predict.__module__.split('.')[0]}' ---")
    print("=========================================================")

    if not os.path.exists(TEST_DATA_FILE):
        print(f"ERROR: Archivo de datos de prueba no encontrado en: {TEST_DATA_FILE}")
        print("Asegúrese de copiar el archivo 'raw_NewPredictChurnData.csv' al directorio '~/test/'.")
        return

    try:
        # 1. Cargar datos de prueba (solo 10 filas para una prueba rápida)
        print(f"-> 1. Cargando datos de prueba desde: {TEST_DATA_FILE}")
        test_data = pd.read_csv(TEST_DATA_FILE).head(10)
    
        if 'PAID_SUBSCRIPTION_CHURN_30D' in test_data.columns:
            test_data = test_data.drop(columns=['PAID_SUBSCRIPTION_CHURN_30D'])

        # 2. Realizar la predicción
        print(f"-> 2. Realizando predicción para {len(test_data)} registros...")
        results = predict(input_data=test_data)

        # 3. Verificar resultados
        if results.get("errors"):
            print("--- ¡ERROR EN LA PREDICCIÓN! ---")
            print(results["errors"])
            return

        # 4. Combinar resultados
        predictions_df = pd.DataFrame({
            'USER_ID': test_data['USER_ID'],
            'SUBSCRIPTION_ID': test_data['SUBSCRIPTION_ID'],
            'PROBABILITY_CHURN': results["probability"],
            'PREDICTION_CHURN': results["prediction"]
        })

        # 5. Guardar el archivo de salida
        predictions_df.to_csv(OUTPUT_FILE_NAME, index=False)
        
        # 6. IMPRIMIR RESULTADOS 
        print("\n=========================================================")
        print("--------- RESULTADOS DE LA PREDICCIÓN -----------")
        print("=========================================================")
        print(predictions_df.to_string(index=False)) 
        print("\n=========================================================")
        print(f"-> 3. PRUEBA EXITOSA. Se generaron {len(predictions_df)} predicciones.")
        print(f"-> 4. Archivo de predicciones guardado en: {OUTPUT_FILE_NAME}")


    except Exception as e:
        print(f"\nERROR CRÍTICO durante la ejecución del test: {e}")
        if "Model assets not loaded" in str(e) or "KeyError" in str(e):
             print("\nADVERTENCIA: Verifique que el paquete .whl que instaló incluya el modelo (.pkl) y el binarizador (.mlb) en su carpeta 'assets'.")


if __name__ == '__main__':
    run_prediction_test()
