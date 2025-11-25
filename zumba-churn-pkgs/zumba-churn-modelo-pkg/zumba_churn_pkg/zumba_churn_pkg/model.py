# zumba_churn_pkg/model.py
import joblib
import pandas as pd
import warnings
import numpy as np 
from typing import Dict, Any, List

# Se importan las variables de ruta y configuración desde config.py
from .config import MODEL_PATH, MLB_PATH, MULTILABEL_FEATURE, SELECTED_FEATURES, TARGET_VARIABLE
from .processing.processor import MultiLabelBinarizerTransformer, pipeline_transformer


# Se inicializan el modelo y los binarizadores globalmente.
_churn_model = None
_mlb_transformer_classes = None 
_expected_features: List[str] = [] # Almacenar la lista de características esperadas

# Se inicializa la bandera de estado para el cargador
_is_loaded = False 

def _load_assets() -> None:
    """Se cargan el modelo y los binarizadores a la memoria global."""
    global _churn_model, _mlb_transformer_classes, _is_loaded, _expected_features
    
    # Se comprueba la existencia de ruta antes de intentar cargar
    if not MODEL_PATH.exists() or not MLB_PATH.exists():
        warnings.warn(
            f"Error de carga de activos (FileNotFound). Se verifica: {MODEL_PATH} y {MLB_PATH}. ¿Se ejecutó el entrenamiento?", 
            UserWarning
        )
        _is_loaded = False
        return

    try:
        # 1. Se carga el modelo XGBoost
        _churn_model = joblib.load(filename=MODEL_PATH)
        
        # 2. Se carga el diccionario de MultiLabelBinarizer ajustados
        _mlb_transformer_classes = joblib.load(filename=MLB_PATH)
        
        # 3. Se obtienen las características esperadas del modelo
        # Esto es crítico para el alineamiento del DataFrame de entrada.
        if hasattr(_churn_model, 'feature_names_in_'):
             _expected_features = _churn_model.feature_names_in_.tolist()
        elif SELECTED_FEATURES:
            _expected_features = SELECTED_FEATURES
        else:
            warnings.warn("No se pudieron obtener los nombres de características del modelo. Usando SELECTED_FEATURES de config.", UserWarning)
            _expected_features = SELECTED_FEATURES
            
        _is_loaded = True
        print("Activos de predicción cargados exitosamente.")

    except Exception as e:
        _is_loaded = False
        warnings.warn(f"Fallo al cargar los activos: {e}", UserWarning)

def make_prediction(*, input_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Función principal para generar predicciones de churn.
    
    Args:
        input_data: DataFrame de Pandas con los datos crudos a predecir.
        
    Returns:
        Un diccionario con las predicciones, probabilidades y errores (si los hay).
    """
    results = {"prediction": None, "probability": None, "errors": None}
    
    # Se realiza la carga de activos si no se ha hecho
    if not _is_loaded:
        _load_assets()
        
    if not _is_loaded or not _churn_model or not _mlb_transformer_classes or not _expected_features:
        results["errors"] = "El modelo no está cargado o los activos están incompletos."
        return results

    try:
        # 1. Se realiza una copia de los datos
        input_data_copy = input_data.copy()

        # 2. Preprocesamiento (Aplica la tubería de transformación)
        mlb_transformer_instance = MultiLabelBinarizerTransformer(
            mlb_all=_mlb_transformer_classes, 
            # CORRECCIÓN CLAVE: Se pasa la lista completa de columnas Multi-Label.
            # 'CLASS_INTENSITY_PREFERENCE' se imputa y binariza junto con la columna configurada.
            list_columns=['CLASS_INTENSITY_PREFERENCE', MULTILABEL_FEATURE] 
        )
        
        processed_data = pipeline_transformer(
            df=input_data_copy,
            mlb_transformer=mlb_transformer_instance,
            selected_features=_expected_features, # Se usa _expected_features para ser más exactos
            multilabel_feature=MULTILABEL_FEATURE,
            target_variable=TARGET_VARIABLE
        )
        
        # 3. Se alinea el DataFrame de datos procesados a las columnas esperadas.
        # .reindex() añade las columnas faltantes (con valor 0) y ordena las existentes.
        processed_data_aligned = processed_data.reindex(
            columns=_expected_features, 
            fill_value=0
        )
        
        # 4. Se asegura que todos los datos sean float (necesario para la mayoría de los modelos ML).
        # El .astype(np.float64) asegura la homogeneidad del tipo de datos.
        processed_data_aligned = processed_data_aligned.astype(np.float64) 
        
        # 5. Se convierte a un array de NumPy (la forma más segura de alimentar a los modelos ML)
        data_for_prediction = processed_data_aligned.values
        
        # 6. Se realiza la Predicción
        prediction_result = _churn_model.predict(data_for_prediction)
        probability_result = _churn_model.predict_proba(data_for_prediction)[:, 1]
        
        # 7. Se formatean los resultados
        results["prediction"] = [int(p) for p in prediction_result]
        results["probability"] = [float(p) for p in probability_result]
        
    except Exception as e:
        # Se retorna un mensaje de error limpio, aunque internamente se podría registrar el traceback.
        error_msg = f"Error durante la predicción: {e}"
        results["errors"] = error_msg
        
    return results
