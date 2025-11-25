# zumba_churn_pkg/__init__.py
from .model import make_prediction

predict = make_prediction  # alias para la API pública

__all__ = ["predict"]
