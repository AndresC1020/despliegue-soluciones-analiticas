# main.py

import uvicorn
from typing import Any
from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from loguru import logger

# Se importan los módulos de la aplicación
from api import api_router
from config import settings, setup_app_logging 

# --- Inicialización del Logging ---
setup_app_logging(config=settings)
logger.info(f"Configuración de Loguru completada. Proyecto: {settings.PROJECT_NAME}")

# --- Inicialización de la Aplicación FastAPI ---
app = FastAPI(
    title=settings.PROJECT_NAME, 
    # Se configura la URL de la documentación OpenAPI
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# --- Router Raíz (Respuesta HTML) ---
root_router = APIRouter()

@root_router.get("/", response_class=HTMLResponse)
def index(request: Request) -> Any:
    """Se proporciona la respuesta HTML básica para la raíz."""
    body = (
        "<html>"
        "<head>"
        f"<title>{settings.PROJECT_NAME}</title>"
        "</head>"
        "<body style='padding: 20px; font-family: sans-serif;'>"
        f"<h1>Bienvenido a la {settings.PROJECT_NAME}</h1>"
        "<div>"
        "Se puede verificar la documentación de la API en: "
        f"<a href='/docs'>{settings.API_V1_STR}/docs</a>"
        "</div>"
        "</body>"
        "</html>"
    )

    return HTMLResponse(content=body)

# --- Inclusión de Routers ---
# Se incluye el Router de la API (con el prefijo /api/v1)
app.include_router(api_router, prefix=settings.API_V1_STR)
# Se incluye el Router Raíz (/)
app.include_router(root_router)


# --- Configuración de CORS ---
if settings.BACKEND_CORS_ORIGINS:
    logger.info("Se configura CORS Middleware...")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def start():
    """
    Se inicia el servidor Uvicorn.
    """
    logger.warning("Se ejecuta en modo de desarrollo. No se recomienda para producción.")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8001,      
        log_level="info" 
    )

if __name__ == "__main__":
    start()
