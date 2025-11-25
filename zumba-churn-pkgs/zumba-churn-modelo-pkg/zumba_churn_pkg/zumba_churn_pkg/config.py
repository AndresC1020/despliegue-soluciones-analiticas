# zumba_churn_pkg/config.py
from pathlib import Path
import yaml
import importlib.resources as pkg_resources
from pydantic import AnyHttpUrl
from pydantic_settings import BaseSettings

PACKAGE_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PACKAGE_ROOT / "assets"

def load_config():
    with pkg_resources.open_text("zumba_churn_pkg", "config.yml") as f:
        return yaml.safe_load(f)

config = load_config()

MODEL_PATH = ASSETS_DIR / config["artifacts"]["model_file"]
MLB_PATH = ASSETS_DIR / config["artifacts"]["binarizer_file"]
MULTILABEL_FEATURE = config["features"]["multilabel_feature"]
SELECTED_FEATURES = config["features"]["selected_features"]
TARGET_VARIABLE = config["data"]["target_variable"]
