# zumba_churn_pkg/train_pipeline.py

import pandas as pd
import joblib
import yaml
import ast
from pathlib import Path

# Modelado
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Variables globales y rutas

# Carpeta raíz del proyecto (sube dos niveles desde este archivo)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Carpeta del paquete
PACKAGE_ROOT = PROJECT_ROOT / "zumba_churn_pkg"

# Archivos de configuración y datos
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATA_DIR = PROJECT_ROOT / "data"
ASSETS_DIR = PACKAGE_ROOT / "assets"
PREPROCESSING_PATH = PACKAGE_ROOT / "processing" / "processor.py"


# --- 1. Funciones Auxiliares para Cargar Configuración y Data ---

def load_config(config_path: Path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_data(file_name: str, data_dir: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_dir / file_name)
        return df
    except FileNotFoundError:
        print(f"Error: Archivo de datos no encontrado en {data_dir / file_name}")
        return pd.DataFrame()

# --- 2. Función de Preprocesamiento y Binarización (Adaptada de preprocessing.py) ---

def fit_and_transform_data(df_train: pd.DataFrame, config: dict):
    df = df_train.copy()
    target_variable = config['data']['target_variable']

    # 1. Eliminar suscripciones anuales
    df = df[df['SUBSCRIPTION_TIER'] != 'Annual']

    # 2. Eliminar columnas irrelevantes
    drop_cols = ['USER_ID','SUBSCRIPTION_ID','NEW_PAID_SUBSCRIPTION_DATE',
                 'PAID_SUBSCRIPTION_DURATION_DAYS','REGION_TYPE_LEVEL_1',
                 'REGION_TYPE_LEVEL_2','SUBSCRIPTION_AFFILIATE','MOTIVATING_FACTORS',
                 'CLASS_FORMAT_PREFERENCE','GENDER_IDENTITY','AGE']
    df = df.drop(columns=drop_cols, errors="ignore")

    # 3. Imputaciones (usando [] para listColumns, 'Unknown' para otros)
    df['CLASS_INTENSITY_PREFERENCE'] = df['CLASS_INTENSITY_PREFERENCE'].fillna('[]')
    df['MUSIC_PREFERENCE'] = df['MUSIC_PREFERENCE'].fillna('[]')
    df['FITNESS_GOAL_PREFERENCE'] = df['FITNESS_GOAL_PREFERENCE'].fillna('Unknown')
    
    # 4. Transformar listas a dummies y guardar el MLB
    listColumns = ['CLASS_INTENSITY_PREFERENCE','MUSIC_PREFERENCE']
    
    mlb_all = {}
    
    for col in listColumns:
        # Convertir strings de lista a listas de Python
        df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        
        mlb = MultiLabelBinarizer()
        # Ajustar y transformar
        dummies = pd.DataFrame(mlb.fit_transform(df[col]), 
                               columns=[f"{col}_{c}" for c in mlb.classes_], 
                               index=df.index)
        
        df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
        mlb_all[col] = mlb

    # 5. Separar X e Y de forma explícita
    if target_variable not in df.columns:
        raise ValueError(f"Variable objetivo '{target_variable}' no encontrada después del preprocesamiento inicial.")

    y = df[target_variable]
    X = df.drop(columns=[target_variable])
    
    # 6. Crear dummies para las variables categóricas restantes
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    X = pd.get_dummies(X, columns=categorical_cols, prefix=categorical_cols, drop_first=True)
    
    # 7. Alineación de columnas con las seleccionadas en config.yml
    final_features = config['features']['selected_features']
    
    # Rellenar con 0 las columnas que faltan
    missing_cols = list(set(final_features) - set(X.columns))
    for col in missing_cols:
        X[col] = 0.0
        
    X = X[final_features]
    
    return X, y, mlb_all

# --- 3. Función Principal de Entrenamiento ---

def run_training():
    
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    
    config = load_config(CONFIG_FILE_PATH)
    
    training_file = config['data']['training_file']
    df_train = load_data(training_file, DATA_DIR)
    
    if df_train.empty:
        return
    
    print(f"Datos cargados. Filas: {len(df_train)}")

    # Preprocesamiento, ajuste del MLB y obtención de X e y
    X, y, mlb_fitted = fit_and_transform_data(df_train, config)
    
    # Separar en conjunto de entrenamiento y validación
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config['model']['hyperparameters']['random_state'], stratify=y
    )

    # 4. Entrenar el modelo XGBoost
    print("Iniciando entrenamiento del modelo...")
    
    model = XGBClassifier(**config['model']['hyperparameters'])
    
    model.fit(X_train, y_train)

    # 5. Evaluar con todas las métricas
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
    
    print("\n--- Métricas del Modelo en Conjunto de Prueba ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("-------------------------------------------------")
    
    # 6. Serializar y guardar el modelo y el MLB

    # Guardar el modelo XGBoost
    model_path = ASSETS_DIR / "churn_model.pkl"
    joblib.dump(model, model_path)
    print(f"Modelo serializado guardado en: {model_path}")
    
    # Guardar el Binarizador MultiLabel ajustado
    mlb_path = ASSETS_DIR / "mlb_classes.pkl"
    joblib.dump(mlb_fitted, mlb_path)
    print(f"Binarizador serializado guardado en: {mlb_path}")
    
    print("\n¡Proceso de entrenamiento completado exitosamente!")

if __name__ == "__main__":
    run_training()
