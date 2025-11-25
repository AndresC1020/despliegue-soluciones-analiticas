# api.py

import pandas as pd
from fastapi import APIRouter, status
from pydantic import BaseModel, Field
from typing import List, Optional, Any
from loguru import logger
from pathlib import Path
import zumba_churn_pkg
import json # Se importa para la conversión de listas a strings JSON

# Se importa la función de predicción
from zumba_churn_pkg.model import make_prediction

# --- Versión del modelo ---
version_file = Path(zumba_churn_pkg.__file__).resolve().parent / "VERSION"
try:
    model_version = version_file.read_text().strip()
except Exception:
    model_version = "0.0.0"

# --- Inicialización del Router ---
api_router = APIRouter()

# --- Función de Adaptación de Datos ---
def _sanitize_multilabel_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte las columnas de lista de Python (procedentes de Pydantic/FastAPI) 
    a strings literales JSON. Esto es necesario porque el transformador MultiLabelBinarizer
    dentro del paquete sellado espera strings para usar ast.literal_eval.
    """
    df_copy = df.copy()
    
    # Columnas multi-etiqueta que deben ser strings literales
    MULTILABEL_COLUMNS = ['MUSIC_PREFERENCE', 'CLASS_INTENSITY_PREFERENCE']
    
    for col in MULTILABEL_COLUMNS:
        if col in df_copy.columns:
            
            def safe_convert(item):
                # Si es una lista de Python (el formato de la API), la convertimos a un string JSON.
                if isinstance(item, list):
                    # json.dumps() genera un string literal válido, ej: '["Pop", "Rock"]'
                    return json.dumps(item)
                
                # Si es nulo (o NaN), se convierte a '[]' como string para que el imputador del paquete lo maneje.
                if item is None or (isinstance(item, float) and pd.isna(item)):
                    return '[]'
                    
                return item # Dejar otros tipos (ej. strings) como están

            df_copy[col] = df_copy[col].apply(safe_convert)
            
    return df_copy
    
# --- Esquemas de entrada (Pydantic) ---
class InputData(BaseModel):
    TRIAL_DURATION_DAYS: float = Field(..., description="Duración de la prueba gratuita en días.")
    SUBSCRIPTION_TIER: str = Field(..., description="Nivel de suscripción (ej: Monthly, Annual).")
    SUBSCRIPTION_SOURCE: str = Field(..., description="Fuente de la suscripción (ej: App Store, Web).")
    SUBSCRIPTION_PROVIDER: str = Field(..., description="Proveedor de pago (ej: apple, google).")
    COUNTRY: str = Field(..., description="País del usuario (ej: USA, Mexico).")
    CLASS_PER_WEEK_GOAL: int = Field(..., description="Meta de clases por semana.")
    DANCE_LEVEL_PREFERENCE: str = Field(..., description="Nivel de danza preferido (ej: Beginner, Advanced).")
    FITNESS_GOAL_PREFERENCE: Optional[str] = Field(None, description="Meta de fitness (ej: lose weight and tone).")
    APP_VERSION: Optional[str] = Field(None, description="Versión de la aplicación.")
    DEVICE_OPERATING_SYSTEM: str = Field(..., description="Sistema operativo del dispositivo (ej: iOS, Android).")
    DEVICE_MANUFACTURER: str = Field(..., description="Fabricante del dispositivo.")
    DEVICE_BRAND: str = Field(..., description="Marca del dispositivo.")
    DEVICE_MODEL: Optional[str] = Field(None, description="Modelo del dispositivo.")
    APP_INSTALL_TO_PAID_SUBSCRIPTION_DAYS: float = Field(..., description="Días entre la instalación y el pago.")
    NO_VIDEO_STARTED_30D: int = Field(..., description="Número de videos NO iniciados en 30 días.")
    NO_VIDEO_WATCHED_30D: int = Field(..., description="Número de videos NO vistos completamente en 30 días.")
    AVG_VIDEO_WATCHED_PERCENTAGE_30D: float = Field(..., description="Porcentaje promedio de video visto en 30 días.")
    AVG_VIDEO_LENGTH_MIN_30D: float = Field(..., description="Duración promedio de videos vistos en 30 días (en minutos).")
    MUSIC_PREFERENCE: List[str] = Field(..., description="Lista de preferencias musicales (ej: [\"Pop\", \"Salsa\"]).")
    CLASS_INTENSITY_PREFERENCE: List[str] = Field(..., description="Lista de intensidades de clase (ej: [\"Medium\"]).")

    # Ejemplo de datos de entrada para la documentación de la API
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "TRIAL_DURATION_DAYS": 7.0,
                    "SUBSCRIPTION_TIER": "Annual",
                    "SUBSCRIPTION_SOURCE": "Web",
                    "SUBSCRIPTION_PROVIDER": "Stripe",
                    "COUNTRY": "United States of America",
                    "CLASS_PER_WEEK_GOAL": 2,
                    "DANCE_LEVEL_PREFERENCE": "Beginner",
                    "FITNESS_GOAL_PREFERENCE": "Start my fitness journey",
                    "APP_VERSION": "3.14.0",
                    "DEVICE_OPERATING_SYSTEM": "Android",
                    "DEVICE_MANUFACTURER": "Google",
                    "DEVICE_BRAND": "Google",
                    "DEVICE_MODEL": "Pixel 8",
                    "APP_INSTALL_TO_PAID_SUBSCRIPTION_DAYS": 30.0,
                    "NO_VIDEO_STARTED_30D": 0,
                    "NO_VIDEO_WATCHED_30D": 0,
                    "AVG_VIDEO_WATCHED_PERCENTAGE_30D": 0.0,
                    "AVG_VIDEO_LENGTH_MIN_30D": 0.0,
                    "MUSIC_PREFERENCE": ["Afro Rhythms", "Pop", "Salsa"],
                    "CLASS_INTENSITY_PREFERENCE": ["Medium"]
                }
            ]
        }
    }

class MultipleInputData(BaseModel):
    inputs: List[InputData]

# --- Endpoints ---

@api_router.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Se verifica el estado de salud de la API."""
    return {"status": "ok", "model_loaded": True}

@api_router.get("/version", status_code=status.HTTP_200_OK)
def version():
    """Se obtiene la versión del modelo."""
    return {"model_version": model_version}

@api_router.post("/predict", status_code=status.HTTP_200_OK)
async def predict(input_data: MultipleInputData):
    """
    Se realizan predicciones de churn en un lote de datos.
    """
    # 1. Se convierte el objeto Pydantic a un DataFrame
    input_df = pd.DataFrame([i.model_dump() for i in input_data.inputs])

    # 2. ADAPTACIÓN: Se sanitizan las columnas multi-etiqueta antes de llamar al paquete.
    input_df_sanitized = _sanitize_multilabel_columns(input_df)

    # 3. Se realiza la predicción con los datos adaptados
    results = make_prediction(input_data=input_df_sanitized)

    if results.get("errors"):
        logger.error(f"Error durante la predicción: {results['errors']}")
        # Se retorna un mensaje claro de error 500
        return {
            "error_detail": results["errors"][0],
            "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR
        }

    # 4. Se retornan los resultados de la predicción
    return results
