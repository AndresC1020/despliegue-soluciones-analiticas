# predict_batch.py

import pandas as pd
from zumba_churn_pkg import config, ASSETS_PATH, OUTPUT_FILE_NAME
from zumba_churn_pkg.model import make_prediction
from pathlib import Path

# Se definen las rutas usando la configuración centralizada
DATA_DIR = Path(__file__).resolve().parent / 'data' # Se asume un directorio 'data' en la raíz
INPUT_FILE_PATH = DATA_DIR / config.config['data']['prediction_file']
OUTPUT_FILE_PATH = DATA_DIR / OUTPUT_FILE_NAME

def run_batch_prediction():
    """
    Ejecuta el pipeline completo de predicción por lotes:
    1. Carga los datos de entrada sin procesar.
    2. Realiza la predicción.
    3. Guarda los resultados en un archivo CSV.
    """
    print("--- Proceso de Predicción por Lotes de Churn ---")
    
    # 1. Validación de archivos
    if not INPUT_FILE_PATH.exists():
        print(f"ERROR: Archivo de entrada no encontrado en: {INPUT_FILE_PATH}")
        return
    
    # Se crea el directorio de datos si no existe
    DATA_DIR.mkdir(exist_ok=True)
    
    try:
        # 2. Carga de datos
        print(f"Cargando datos desde: {INPUT_FILE_PATH}")
        input_data = pd.read_csv(INPUT_FILE_PATH)

        # Se guarda el ID original si existe para el mapeo
        original_indices = input_data.index.copy() 

        # 3. Predicción
        print("Realizando predicciones...")
        results = make_prediction(input_data=input_data)
        
        if results["errors"]:
            print(f"ERROR durante la predicción: {results['errors']}")
            return

        # 4. Creación del DataFrame de salida
        predictions_df = pd.DataFrame({
            'original_index': original_indices,
            'Churn_Probability': results["probability"],
            'Churn_Predicted_Class': results["prediction"]
        })
        
        # 5. Guardado de resultados
        predictions_df.to_csv(OUTPUT_FILE_PATH, index=False)
        
        print("-" * 40)
        print(f"Predicciones generadas exitosamente para {len(predictions_df)} registros.")
        print(f"Archivo guardado en: {OUTPUT_FILE_PATH}")
        print(f"Versión del modelo usada: {config.__version__}")
        print("-" * 40)

    except Exception as e:
        print(f"Ocurrió un error inesperado durante el pipeline: {e}")

if __name__ == "__main__":
    run_batch_prediction()
