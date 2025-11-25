# api.py

import uvicorn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional, Any
# Se importa la función de predicción y la versión del paquete desde el __init__.py
from zumba_churn_pkg import predict
from zumba_churn_pkg import __version__ as model_version 

# Inicializar la aplicación FastAPI
app = FastAPI(
    title="Zumba Churn Predictor API",
    description="API para predecir la probabilidad de abandono (churn) de un cliente.",
    version=model_version
)

# Definición del esquema de datos de entrada usando Pydantic
class InputData(BaseModel):
    TRIAL_DURATION_DAYS: float = Field(..., description="Duración de la prueba gratuita en días.")
    SUBSCRIPTION_TIER: str = Field(..., description="Nivel de suscripción (ej: Monthly, Annual).")
    SUBSCRIPTION_SOURCE: str = Field(..., description="Fuente de la suscripción (ej: App Store, Web).")
    SUBSCRIPTION_PROVIDER: str = Field(..., description="Proveedor de pago (ej: apple, google).")
    COUNTRY: str = Field(..., description="País del usuario (ej: USA, Mexico).")
    CLASS_PER_WEEK_GOAL: int = Field(..., description="Meta de clases por semana.")
    DANCE_LEVEL_PREFERENCE: str = Field(..., description="Nivel de baile preferido.")
    FITNESS_GOAL_PREFERENCE: str = Field(..., description="Objetivo fitness (ej: lose weight and tone).")
    APP_VERSION: str = Field(..., description="Versión de la aplicación.")
    DEVICE_OPERATING_SYSTEM: str = Field(..., description="Sistema operativo del dispositivo.")
    DEVICE_MANUFACTURER: str = Field(..., description="Fabricante del dispositivo.")
    DEVICE_BRAND: str = Field(..., description="Marca del dispositivo.")
    DEVICE_MODEL: str = Field(..., description="Modelo del dispositivo.")
    APP_INSTALL_TO_PAID_SUBSCRIPTION_DAYS: float = Field(..., description="Días entre instalación y suscripción de pago.")
    NO_VIDEO_STARTED_30D: int = Field(..., description="Número de videos no iniciados en los últimos 30 días.")
    NO_VIDEO_WATCHED_30D: int = Field(..., description="Número de videos no vistos completamente en los últimos 30 días.")
    AVG_VIDEO_WATCHED_PERCENTAGE_30D: float = Field(..., description="Porcentaje promedio de video visto en los últimos 30 días.")
    AVG_VIDEO_LENGTH_MIN_30D: float = Field(..., description="Duración promedio de video en minutos en los últimos 30 días.")
    MUSIC_PREFERENCE: Optional[Any] = Field(None, description="Preferencia musical (puede ser lista de strings o string 'Pop, Rock').")

# Esquema para manejar una lista de inputs
class MultipleInputData(BaseModel):
    inputs: List[InputData]

# Endpoint de salud
@app.get("/health", status_code=200)
def health_check():
    """Endpoint para verificar la salud de la API."""
    return {"status": "ok", "model_loaded": True}

# Endpoint de predicción
@app.post("/predict", status_code=200)
async def predict_endpoint(input_data: MultipleInputData):
    """
    Se realiza la predicción de churn en un lote de datos.
    """
    # Se convierten los datos de Pydantic a un DataFrame de Pandas
    input_df = pd.DataFrame([i.model_dump() for i in input_data.inputs])
    
    # Se realiza la predicción usando la función del paquete
    results = predict(input_data=input_df)
    
    if results["errors"] is not None:
        # Se retorna el error si algo falla en el proceso de predicción
        return {"error": results["errors"], "status": "error"}

    return {
        "predictions": results["prediction"],
        "probabilities": results["probability"],
        "version": model_version
    }
