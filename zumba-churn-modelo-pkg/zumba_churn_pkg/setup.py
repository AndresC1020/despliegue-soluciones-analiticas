import io
from pathlib import Path
from setuptools import find_packages, setup

# --- 1. Definición de Rutas Base ---
ROOT_DIR = Path(__file__).resolve().parent
REQUIREMENTS_DIR = ROOT_DIR / "requirements"
PACKAGE_DIR = ROOT_DIR / "zumba_churn_pkg"

# --- 2. Funciones Auxiliares ---
def get_version():
    """Lee el archivo VERSION dentro del paquete para obtener la versión."""
    try:
        with open(PACKAGE_DIR / "VERSION") as version_file:
            return version_file.read().strip()
    except FileNotFoundError:
        return "0.0.0"  # Fallback si no se encuentra

def get_requirements(filename: str) -> list:
    """Devuelve los requisitos de un archivo de requisitos en el directorio 'requirements'."""
    requirements_path = REQUIREMENTS_DIR / filename
    try:
        with io.open(requirements_path, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        print(f"Advertencia: Archivo de requisitos no encontrado en {requirements_path}.")
        return []

# --- 3. Obtención de Requisitos de Producción ---
required = get_requirements("requirements.txt")

# --- 4. Configuración del Setup ---
setup(
    name="zumba_churn_pkg",
    version=get_version(),
    description="Modelo XGBoost para predicción de churn",
    author="Daniela Uscategui, Andres Alfonso, Francisco Amorocho",
    license="MIT",
    packages=find_packages(exclude=("tests",)),
    install_requires=required,
    include_package_data=True,
    package_data={
        "zumba_churn_pkg": [
            "assets/*.pkl",
            "config.yml",
            "VERSION",
        ]
    },
    zip_safe=False,
)
