#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de versionamiento para registrar el Modelo XGBoost en MLflow, 
"""

# Se importan las librerías necesarias.
import ast
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from mlflow.models.signature import infer_signature
from mlflow.data.pandas_dataset import PandasDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import sys
import warnings

# Silenciar warnings específicos de MLflow
warnings.filterwarnings('ignore', category=FutureWarning, module='mlflow.tracking')
warnings.filterwarnings('ignore', category=UserWarning, module='mlflow.data')
warnings.filterwarnings('ignore', category=UserWarning, module='mlflow.types.utils')


# --- CONFIGURACIÓN DE HIPERPARÁMETROS ---

PARAMS = {
    'colsample_bytree': 0.6, 
    'learning_rate': 0.035, 
    'max_depth': 5, 
    'min_child_weight': 9, 
    'n_estimators': 300, 
    'reg_lambda': 8, 
    'subsample': 0.9,                  
    
    # Parámetros fijos necesarios para la configuración de XGBoost.
    'enable_categorical': True,
    'eval_metric': 'auc',
    'random_state': 0
}

# --- CONFIGURACIÓN DE MLFLOW ---
# Se establece o se crea el experimento.
experiment = mlflow.set_experiment("XGBoost_Churn_Deployment")


# --- 1. PREPROCESAMIENTO DE DATOS ---

try:
    churnData = pd.read_csv('ZumbaConsumerApp_Churn30D.csv')
except FileNotFoundError:
    print("Error: Se requiere el archivo 'ZumbaConsumerApp_Churn30D.csv' en el directorio de ejecución.")
    sys.exit(1)

# Se filtra, limpian y transforman los datos.
churnData = churnData[churnData['SUBSCRIPTION_TIER'] != 'Annual']
churnData = churnData.drop(columns = ['USER_ID', 'SUBSCRIPTION_ID', 'NEW_PAID_SUBSCRIPTION_DATE', 'PAID_SUBSCRIPTION_DURATION_DAYS', 'REGION_TYPE_LEVEL_1', 'REGION_TYPE_LEVEL_2', 'SUBSCRIPTION_AFFILIATE', 'MOTIVATING_FACTORS', 'CLASS_FORMAT_PREFERENCE', 'GENDER_IDENTITY', 'AGE'])
churnData = churnData.dropna(subset = ['DEVICE_BRAND', 'DEVICE_OPERATING_SYSTEM', 'CLASS_INTENSITY_PREFERENCE', 'DANCE_LEVEL_PREFERENCE'])
churnData['CLASS_INTENSITY_PREFERENCE'] = churnData['CLASS_INTENSITY_PREFERENCE'].fillna('[]')
churnData['MUSIC_PREFERENCE'] = churnData['MUSIC_PREFERENCE'].fillna('[]')
churnData['FITNESS_GOAL_PREFERENCE'] = churnData['FITNESS_GOAL_PREFERENCE'].fillna('Unknown')

listColumns = ['CLASS_INTENSITY_PREFERENCE', 'MUSIC_PREFERENCE']
for col in listColumns:
    churnData[col] = churnData[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

mlb = MultiLabelBinarizer()
for col in listColumns:
    dummies = pd.DataFrame(mlb.fit_transform(churnData[col]), columns = [f'{col}_{cl}' for cl in mlb.classes_], index = churnData.index)
    churnData = pd.concat([churnData, dummies], axis = 1)
churnData = churnData.drop(columns = listColumns)

categoricalColumns = [
    'SUBSCRIPTION_TIER', 'SUBSCRIPTION_SOURCE', 'SUBSCRIPTION_PROVIDER', 'COUNTRY',
    'DANCE_LEVEL_PREFERENCE', 'FITNESS_GOAL_PREFERENCE', 'APP_VERSION',
    'DEVICE_OPERATING_SYSTEM', 'DEVICE_MANUFACTURER', 'DEVICE_BRAND', 'DEVICE_MODEL'
    ]
churnData[categoricalColumns] = churnData[categoricalColumns].astype('category')

variablesModelo = ['TRIAL_DURATION_DAYS', 'SUBSCRIPTION_TIER', 'SUBSCRIPTION_SOURCE', 'SUBSCRIPTION_PROVIDER', 'COUNTRY', 'CLASS_PER_WEEK_GOAL', 'DANCE_LEVEL_PREFERENCE', 'FITNESS_GOAL_PREFERENCE', 'APP_VERSION', 'DEVICE_OPERATING_SYSTEM', 'DEVICE_BRAND', 'APP_INSTALL_TO_PAID_SUBSCRIPTION_DAYS', 'NO_VIDEO_STARTED_30D', 'NO_VIDEO_WATCHED_30D', 'AVG_VIDEO_WATCHED_PERCENTAGE_30D', 'AVG_VIDEO_LENGTH_MIN_30D', 'CLASS_INTENSITY_PREFERENCE_High', 'CLASS_INTENSITY_PREFERENCE_Low', 'CLASS_INTENSITY_PREFERENCE_Medium', 'MUSIC_PREFERENCE_Afro Rhythms', 'MUSIC_PREFERENCE_Afrobeats', 'MUSIC_PREFERENCE_Alternative', 'MUSIC_PREFERENCE_Bachata', 'MUSIC_PREFERENCE_Belly Dance', 'MUSIC_PREFERENCE_Bellydance', 'MUSIC_PREFERENCE_Bhangra', 'MUSIC_PREFERENCE_Blues', 'MUSIC_PREFERENCE_Bollywood', 'MUSIC_PREFERENCE_Brazilian Rhythms', 'MUSIC_PREFERENCE_Broadway', 'MUSIC_PREFERENCE_Caribbean Rhythms', 'MUSIC_PREFERENCE_Chill Out', 'MUSIC_PREFERENCE_Country', 'MUSIC_PREFERENCE_Cumbia', 'MUSIC_PREFERENCE_Disco', 'MUSIC_PREFERENCE_Electronic', 'MUSIC_PREFERENCE_House', 'MUSIC_PREFERENCE_K-Pop', 'MUSIC_PREFERENCE_Merengue', 'MUSIC_PREFERENCE_Other', 'MUSIC_PREFERENCE_Pop', 'MUSIC_PREFERENCE_R&B', 'MUSIC_PREFERENCE_Reggae', 'MUSIC_PREFERENCE_Reggaeton', 'MUSIC_PREFERENCE_Rock', 'MUSIC_PREFERENCE_Salsa', 'MUSIC_PREFERENCE_Soca', 'MUSIC_PREFERENCE_Techno', 'MUSIC_PREFERENCE_World Rhythms']

xTotal = churnData[variablesModelo]
yTotal = churnData['PAID_SUBSCRIPTION_CHURN_30D']
xTrain, xTest, yTrain, yTest = train_test_split(xTotal, yTotal, test_size = 0.3, random_state = 0, stratify = yTotal)


# --- 2. MODELO Y REGISTRO ---

with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="XGBoost_Model"):
    
    # Se registra el conjunto de datos de entrenamiento
    train_df = pd.concat([xTrain, yTrain], axis=1)
    train_dataset = mlflow.data.from_pandas(
        train_df,
        source="ZumbaConsumerApp_Churn30D.csv",
        name="ChurnTrainingData"
    )
    
    mlflow.log_input(train_dataset, context="training") 

    # Se inicializa y entrena el modelo.
    xgbModel = xgb.XGBClassifier(**PARAMS)
    xgbModel.fit(xTrain, yTrain)

    # Firma del modelo
    signature = infer_signature(xTrain, xgbModel.predict(xTrain))
    
    # Se realizan predicciones y se calculan las métricas.
    yPred = xgbModel.predict(xTest)
    yPredProb = xgbModel.predict_proba(xTest)[:, 1]

    # Se calculan las métricas de evaluación.
    modelAUC = roc_auc_score(yTest, yPredProb)
    modelAccuracy = accuracy_score(yTest, yPred)
    modelPrecision = precision_score(yTest, yPred, zero_division=0)
    modelRecall = recall_score(yTest, yPred)
    modelF1 = f1_score(yTest, yPred)

    # Se registran Parámetros, Métricas y el Modelo 
    mlflow.log_params(xgbModel.get_params())
    mlflow.log_metric("AUC", modelAUC)
    mlflow.log_metric("Accuracy", modelAccuracy)
    mlflow.log_metric("Precision", modelPrecision)
    mlflow.log_metric("Recall", modelRecall)
    mlflow.log_metric("F1_Score", modelF1)
    
    # Se registra el modelo con la firma 
    mlflow.xgboost.log_model(xgbModel, name="churn_xgboost_model", signature=signature)
    
    # Se imprimen los resultados en la terminal.
    print("\n--- Resultados Modelo ---")
    print(f'ROC AUC: {modelAUC:.4f}')
    print(f'Accuracy: {modelAccuracy:.4f}')
    print(f'Precision: {modelPrecision:.4f}')
    print(f'Recall: {modelRecall:.4f}')
    print(f'F1 Score: {modelF1:.4f}')
    print(f"\n[MLflow] Modelo Base registrado. AUC: {modelAUC:.4f}")
