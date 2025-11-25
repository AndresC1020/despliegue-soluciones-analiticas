# processing/processor.py

import pandas as pd
import ast
from sklearn.preprocessing import MultiLabelBinarizer

# --- Clases de Transformación ---

class MultiLabelBinarizerTransformer:
    """
    Aplicación del MultiLabelBinarizer (MLB) entrenado a la columna 'MUSIC_PREFERENCE'.
    Se asume que la columna contiene una lista de strings o un string evaluable como lista.
    """
    def __init__(self, mlb: MultiLabelBinarizer):
        # Se recibe el objeto MLB ya entrenado
        self.mlb = mlb
        self.feature_name = 'MUSIC_PREFERENCE'

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Aplica el MLB y fusiona las nuevas columnas one-hot-encoded al DataFrame."""
        X_copy = X.copy()
        
        # 1. Función interna para convertir datos a lista de strings
        def safe_eval(x):
            try:
                # Se intenta evaluar el string como una lista
                if isinstance(x, str) and (x.startswith('[') and x.endswith(']')):
                    return ast.literal_eval(x)
                # Si es un solo string no nulo, se envuelve en una lista
                elif pd.notna(x) and isinstance(x, str):
                    return [x.strip()]
                return []
            except (ValueError, SyntaxError):
                return []
        
        X_copy[self.feature_name] = X_copy[self.feature_name].apply(safe_eval)
        
        # 2. Se aplica la transformación MLB
        transformed_data = self.mlb.transform(X_copy[self.feature_name].tolist())
        
        # 3. Creación del DataFrame con las nuevas columnas
        mlb_df = pd.DataFrame(
            transformed_data,
            columns=[f'{self.feature_name}_{c}' for c in self.mlb.classes_],
            index=X_copy.index
        )
        
        # 4. Se elimina la columna original y se concatenan los resultados
        X_copy = X_copy.drop(columns=[self.feature_name])
        X_copy = pd.concat([X_copy.reset_index(drop=True), mlb_df.reset_index(drop=True)], axis=1)

        return X_copy

class FeatureSelector:
    """Selecciona las características finales requeridas por el modelo."""
    def __init__(self, features_to_keep: list):
        self.features_to_keep = features_to_keep

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Asegura la existencia y el orden correcto de las características.
        Rellena con 0 las columnas que falten.
        """
        X_copy = X.copy()
        
        # Se rellena con 0 las columnas faltantes (variables dummy/OHE)
        for col in self.features_to_keep:
            if col not in X_copy.columns:
                X_copy[col] = 0.0
                
        # Se asegura que el orden de las columnas sea el mismo que el del entrenamiento
        return X_copy[self.features_to_keep]

# --- Función de Preprocesamiento Completa ---

def preprocess_for_model(df: pd.DataFrame, mlb: MultiLabelBinarizer, features_to_keep: list) -> pd.DataFrame:
    """
    Aplica la secuencia completa de preprocesamiento necesaria para la predicción.
    """
    df_processed = df.copy()

    # 1. Aplicación del transformador MultiLabelBinarizer
    mlb_transformer = MultiLabelBinarizerTransformer(mlb=mlb)
    df_processed = mlb_transformer.transform(df_processed)
    
    # 2. Aplicación del FeatureSelector para alinear las columnas
    feature_selector = FeatureSelector(features_to_keep=features_to_keep)
    df_final = feature_selector.transform(df_processed)
    
    return df_final
