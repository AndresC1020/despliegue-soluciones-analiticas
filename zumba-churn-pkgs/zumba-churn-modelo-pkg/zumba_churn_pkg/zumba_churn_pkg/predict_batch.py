# predict_batch.py

import pandas as pd
import os
import warnings

# Se importa la función de predicción expuesta en el __init__ del paquete
from .. import predict

# --- Configuración ---
# La ruta a los datos se ajusta asumiendo que se ejecuta desde la raíz del proyecto
DATA_FILE = "../data/raw_NewPredictChurnData.csv" 
OUTPUT_FILE_NAME = "predictions_output.csv"

warnings.filterwarnings("ignore")

def run_prediction_batch():
    """
    Se carga un lote de datos, se realiza la predicción y se guarda el resultado.
    """
    if not os.path.exists(DATA_FILE):
        print(f"Error: No se encontró el archivo de datos: {DATA_FILE}")
        return

    try:
        # 1. Se cargan los datos
        input_data = pd.read_csv(DATA_FILE)
        print(f"Se cargaron {len(input_data)} registros para predicción.")
        
        # 2. Se realiza la predicción
        results = predict(input_data=input_data)
        
        if results["errors"] is not None:
            print(f"Error durante la predicción: {results['errors']}")
            return

        # 3. Se añade el resultado al DataFrame y se guarda
        input_data['churn_prediction'] = results["prediction"]
        input_data['churn_probability'] = results["probability"]
        
        input_data.to_csv(OUTPUT_FILE_NAME, index=False)
        print(f"Predicciones guardadas en {OUTPUT_FILE_NAME}")

    except Exception as e:
        print(f"Ocurrió un error al ejecutar el lote de predicción: {e}")
        

if __name__ == '__main__':
    run_prediction_batch()
