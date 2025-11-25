# zumba_churn_pkg/processing/processor.py

import pandas as pd
import numpy as np
import ast 
from typing import Dict, List
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer

from ..config import (
    SELECTED_FEATURES, 
    MULTILABEL_FEATURE
)

# --- 1. MultiLabelBinarizerTransformer ---

class MultiLabelBinarizerTransformer(BaseEstimator, TransformerMixin):
    """
    Se aplica MultiLabelBinarizer a las columnas de tipo lista/multietiqueta.
    Se utiliza binarizadores preajustados ('mlb_all') para la predicción.
    """
    def __init__(self, mlb_all: Dict[str, MultiLabelBinarizer], list_columns: List[str]):
        """
        Args:
            mlb_all: Diccionario de MultiLabelBinarizer ajustados.
            list_columns: Lista de nombres de columnas a binarizar.
        """
        self.mlb_all = mlb_all
        self.list_columns = list_columns
        
    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        
        for col in self.list_columns:
            def convert_to_list(item):
                if pd.isna(item) or item == '[]' or item == '':
                    return []
                try:
                    # Se intenta evaluar como lista literal
                    return ast.literal_eval(item)
                except (ValueError, TypeError, SyntaxError):
                    # Si falla, se intenta dividir por coma
                    if isinstance(item, str):
                        return [x.strip() for x in item.split(',') if x.strip()]
                    return []
            
            X_copy[col] = X_copy[col].apply(convert_to_list)

            if col in self.mlb_all:
                mlb = self.mlb_all[col]
                # Se crea un DataFrame de las columnas binarizadas
                binarized_data = mlb.transform(X_copy[col])
                df_bin = pd.DataFrame(
                    binarized_data,
                    index=X_copy.index,
                    # Se utiliza el nombre de la columna original como prefijo
                    columns=[f"{col}_{cls}" for cls in mlb.classes_] 
                )
                # Se concatena y se elimina la columna original
                X_copy = pd.concat([X_copy.drop(columns=[col]), df_bin], axis=1)
                
        return X_copy


# --- 2. FeatureSelector ---

class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Se selecciona un subconjunto de características especificadas.
    """
    def __init__(self, features_to_select: List[str]):
        self.features_to_select = features_to_select
        
    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X) 
        cols_to_select = [col for col in self.features_to_select if col in X.columns]
        
        return X[cols_to_select]


# --- 3. Pipeline Principal de Transformación de Datos ---

def pipeline_transformer(
    *, 
    df: pd.DataFrame, 
    mlb_transformer: MultiLabelBinarizerTransformer, 
    selected_features: List[str],
    multilabel_feature: str,
    target_variable: str 
) -> pd.DataFrame:
    """
    Se aplica la secuencia de preprocesamiento para transformar los datos crudos
    en la forma final lista para el modelo.
    """
    df_processed = df.copy()
    
    # --- PASO 1: Filtrar Columnas Crudas ---
    # Se filtran solo las columnas crudas necesarias
    raw_features_to_keep = [
        f for f in selected_features if f not in [
            'CLASS_INTENSITY_PREFERENCE_High', 
            'CLASS_INTENSITY_PREFERENCE_Low'
        ] and not f.startswith(f"{multilabel_feature}_") 
    ]
    
    # Se agregan las columnas crudas de las que dependen las features finales
    required_raw_cols = list(set(raw_features_to_keep + ['CLASS_INTENSITY_PREFERENCE', 'FITNESS_GOAL_PREFERENCE', multilabel_feature]))
    
    # Se seleccionan solo las columnas crudas relevantes para la transformación
    cols_to_select = [col for col in required_raw_cols if col in df_processed.columns]
    df_processed = df_processed[cols_to_select]


    # --- PASO 2: Imputaciones ---
    
    # Se imputa para CLASS_INTENSITY_PREFERENCE y MUSIC_PREFERENCE 
    if 'CLASS_INTENSITY_PREFERENCE' in df_processed.columns:
        df_processed['CLASS_INTENSITY_PREFERENCE'] = df_processed['CLASS_INTENSITY_PREFERENCE'].fillna('[]')
        
    if multilabel_feature in df_processed.columns:
        df_processed[multilabel_feature] = df_processed[multilabel_feature].fillna('[]')
    
    # Se imputa para otras categóricas (ej: 'Unknown')
    if 'FITNESS_GOAL_PREFERENCE' in df_processed.columns:
        df_processed['FITNESS_GOAL_PREFERENCE'] = df_processed['FITNESS_GOAL_PREFERENCE'].fillna('Unknown')
    
    
    # --- PASO 3: Binarización Multi-Label ---
    df_processed = mlb_transformer.transform(df_processed)
    
    
    # --- PASO 4: One-Hot Encoding de Variables Categóricas Restantes ---
    
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    
    # Se elimina la columna TARGET si está presente 
    if target_variable in categorical_cols:
        categorical_cols.remove(target_variable)

    # Se aplica One-Hot Encoding
    df_processed = pd.get_dummies(
        df_processed, 
        columns=categorical_cols, 
        prefix=categorical_cols, 
        dummy_na=False 
    )

    # --- PASO 5: Eliminar columna Target si existe ---
    if target_variable in df_processed.columns:
        df_processed = df_processed.drop(columns=[target_variable])
    
    return df_processed
