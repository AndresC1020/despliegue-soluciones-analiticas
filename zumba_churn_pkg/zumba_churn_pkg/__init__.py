# zumba_churn_pkg/__init__.py

import os
import yaml
from pathlib import Path
from zumba_churn_pkg.processing.processor import MultiLabelBinarizerTransformer, FeatureSelector

# Ruta base del proyecto (un nivel arriba de zumba_churn_pkg)
PACKAGE_ROOT = Path(__file__).parent.parent
# Ruta al archivo de configuración
CONFIG_FILE_PATH = PACKAGE_ROOT / 'config.yml'

# Carga la versión del paquete
with open(PACKAGE_ROOT / 'VERSION') as version_file:
    __version__ = version_file.read().strip()

# Función para cargar el config.yml
def load_config(config_path: Path):
    """Carga la configuración desde config.yml."""
    if not config_path.exists():
        raise FileNotFoundError(f"Archivo de configuración no encontrado en: {config_path}")
        
    with open(config_path, 'r') as conf_file:
        parsed_config = yaml.safe_load(conf_file.read())
    return parsed_config

config = load_config(CONFIG_FILE_PATH)

# Rutas a los activos del modelo
ASSETS_PATH = PACKAGE_ROOT / config['artifacts']['assets_path']
MODEL_FILE_NAME = config['artifacts']['model_file']
MLB_FILE_NAME = config['artifacts']['binarizer_file']

# Definir variables 
__all__ = ['MultiLabelBinarizerTransformer', 'FeatureSelector', 'config', '__version__']
