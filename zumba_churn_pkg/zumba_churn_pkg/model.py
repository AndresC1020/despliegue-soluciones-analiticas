# zumba_churn_pkg/model.py

import joblib
import pandas as pd
# Se importa config desde el __init__ del paquete
from zumba_churn_pkg import config
# Se importa la función de preprocesamiento
from zumba_churn_pkg.processing.processor import preprocess_for_model
from sklearn.preprocessing import MultiLabelBinarizer
import xgboost as xgb

# --- Carga de Activos ---

def load_model_assets():
    """Carga el modelo y el MultiLabelBinarizer desde los archivos .pkl."""
    
    model_path = config.ASSETS_PATH / config.MODEL_FILE_NAME
    mlb_path = config.ASSETS_PATH / config.MLB_FILE_NAME
    
    if not model_path.exists() or not mlb_path.exists():
        # Se lanza un error si faltan archivos, indicando la ruta esperada
        raise FileNotFoundError(
            f"Faltan archivos de activos del modelo. Ruta esperada: {config.ASSETS_PATH}"
        )

    # Carga el modelo XGBoost
    model = joblib.load(model_path)
    # Carga el MultiLabelBinarizer
    mlb = joblib.load(mlb_path)
    
    return model, mlb

# --- Función Principal de Predicción ---

def make_prediction(input_data: pd.DataFrame) -> dict:
    """
    Realiza predicciones usando el modelo entrenado.
    
    Args:
        input_data: DataFrame de Pandas con los datos de entrada sin procesar.

    Returns:
        Un diccionario con las predicciones, probabilidades y cualquier error.
    """
    results = {"prediction": None, "probability": None, "errors": None}
    
    try:
        # 1. Carga de activos
        model, mlb = load_model_assets()
        
        # 2. Se obtiene la lista de features desde el archivo de configuración
        features_to_keep = config.config['features']['selected_features']

        # 3. Preprocesamiento de los datos de entrada
        data_processed = preprocess_for_model(
            df=input_data, 
            mlb=mlb, 
            features_to_keep=features_to_keep
        )
        
        # 4. Predicción
        predictions = model.predict(data_processed)
        probabilities = model.predict_proba(data_processed)[:, 1] # Probabilidad de churn (clase 1)

        # 5. Se formatean los resultados
        results["prediction"] = [int(p) for p in predictions]
        results["probability"] = [float(p) for p in probabilities]
        results["errors"] = None

    except Exception as e:
        # Se captura cualquier error para retornarlo como parte del resultado
        results["errors"] = str(e)
        
    return results
