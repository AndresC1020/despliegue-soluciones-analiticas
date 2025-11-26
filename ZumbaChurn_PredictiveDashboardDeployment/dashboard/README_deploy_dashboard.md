# Manual de Despliegue del Dashboard: Zumba Predicción Abandono

Este documento describe cómo instalar y desplegar el dashboard de predicción de churn
desarrollado en Dash (Python).

## 1. Requisitos

- Python 3.11+
- pip
- Docker
- Archivo de datos: `data/predictions_ChurnPredictedData.csv`

El archivo `app_dashboard.py` asume que el CSV se encuentra en la ruta:
`./data/predictions_ChurnPredictedData.csv`.

---

## 2. Ejecución local sin Docker

1. Crear y activar un entorno virtual:

```bash
cd ZumbaChurn_PredictiveDashboardDeployment/dashboard

python -m venv venv
# En Windows:
venv\Scripts\activate
# En macOS / Linux:
# source venv/bin/activate