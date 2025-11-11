import dash
from dash import dcc
from dash import html
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import ast
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import warnings
from dash.exceptions import PreventUpdate
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- CONFIGURACIÓN DE CONSTANTES Y PARÁMETROS ÓPTIMOS ---
ZUMBA_PINK = '#ff005c'
MODEL_BLUE = '#00c7e5'
HIGH_RISK_RED = '#ef4444'
PURPLE = '#7678ed'

# Mejores hiperparámetros predefinidos (se obtienen del tunning)
BEST_PARAMS = {
    'colsample_bytree': 0.6,
    'learning_rate': 0.035,
    'max_depth': 5,
    'min_child_weight': 9,
    'n_estimators': 300,
    'reg_lambda': 8,
    'subsample': 0.9
}


# ==============================================================================
# 1. RESULTADOS DEL MODELO
# ==============================================================================

def get_model_results():
    """
    Se ejecuta carga, preprocesamiento y entrenamiento del modelo XGBoost utilizando los hiperparámetros pre-optimizados
    para generar el diccionario de métricas y datos para el dashboard.
    """
    print("--- 1. Iniciando Pipeline ML: Carga, Preprocesamiento y Entrenamiento ---")

    # 1. CARGA Y PREPROCESAMIENTO DE DATOS
    churnData = pd.read_csv('ZumbaConsumerApp_Churn30D.csv')

    
    # Se realiza el filtrado y se eliminan columnas no deseadas.
    churnData = churnData[churnData['SUBSCRIPTION_TIER'] != 'Annual']
    churnData = churnData.drop(columns = ['USER_ID', 'SUBSCRIPTION_ID', 'NEW_PAID_SUBSCRIPTION_DATE', 
                                          'PAID_SUBSCRIPTION_DURATION_DAYS', 'REGION_TYPE_LEVEL_1', 
                                          'REGION_TYPE_LEVEL_2', 'SUBSCRIPTION_AFFILIATE', 
                                          'MOTIVATING_FACTORS', 'CLASS_FORMAT_PREFERENCE', 
                                          'GENDER_IDENTITY', 'AGE', 'DEVICE_MANUFACTURER', 'DEVICE_MODEL'])
    
    # Se manejan valores faltantes.
    churnData = churnData.dropna(subset = ['DEVICE_BRAND', 'DEVICE_OPERATING_SYSTEM', 
                                          'CLASS_INTENSITY_PREFERENCE', 'DANCE_LEVEL_PREFERENCE'])
    churnData['CLASS_INTENSITY_PREFERENCE'] = churnData['CLASS_INTENSITY_PREFERENCE'].fillna('[]')
    churnData['MUSIC_PREFERENCE'] = churnData['MUSIC_PREFERENCE'].fillna('[]')
    churnData['FITNESS_GOAL_PREFERENCE'] = churnData['FITNESS_GOAL_PREFERENCE'].fillna('Unknown')
    
    # Se aplica MultiLabelBinarizer para codificar características de lista (multiselect).
    listColumns = ['CLASS_INTENSITY_PREFERENCE', 'MUSIC_PREFERENCE']
    for col in listColumns:
        churnData[col] = churnData[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    mlb = MultiLabelBinarizer()
    for col in listColumns:
        dummies = pd.DataFrame(mlb.fit_transform(churnData[col]), 
                               columns = [f'{col}_{cl}' for cl in mlb.classes_], 
                               index = churnData.index)
        churnData = pd.concat([churnData, dummies, ], axis = 1)
    
    churnData = churnData.drop(columns = listColumns)

    # Se establece el tipo 'category' para las columnas categóricas.
    categoricalColumns = [
        'SUBSCRIPTION_TIER', 'SUBSCRIPTION_SOURCE', 'SUBSCRIPTION_PROVIDER', 
        'COUNTRY', 'DANCE_LEVEL_PREFERENCE', 'FITNESS_GOAL_PREFERENCE', 
        'APP_VERSION', 'DEVICE_OPERATING_SYSTEM', 'DEVICE_BRAND'
    ]
    churnData[categoricalColumns] = churnData[categoricalColumns].astype('category')
    
    # 2. SELECCIÓN DE VARIABLES Y SPLIT
    print("--- 2. Selección de Variables ---")
    variablesModelo = [
        'TRIAL_DURATION_DAYS', 'SUBSCRIPTION_TIER', 'SUBSCRIPTION_SOURCE', 'SUBSCRIPTION_PROVIDER', 
        'COUNTRY', 'CLASS_PER_WEEK_GOAL', 'DANCE_LEVEL_PREFERENCE', 'FITNESS_GOAL_PREFERENCE', 
        'APP_VERSION', 'DEVICE_OPERATING_SYSTEM', 'DEVICE_BRAND', 'APP_INSTALL_TO_PAID_SUBSCRIPTION_DAYS', 
        'NO_VIDEO_STARTED_30D', 'NO_VIDEO_WATCHED_30D', 'AVG_VIDEO_WATCHED_PERCENTAGE_30D', 
        'AVG_VIDEO_LENGTH_MIN_30D', 'CLASS_INTENSITY_PREFERENCE_High', 'CLASS_INTENSITY_PREFERENCE_Low', 
        'CLASS_INTENSITY_PREFERENCE_Medium', 'MUSIC_PREFERENCE_Afro Rhythms', 'MUSIC_PREFERENCE_Afrobeats', 
        'MUSIC_PREFERENCE_Alternative', 'MUSIC_PREFERENCE_Bachata', 'MUSIC_PREFERENCE_Belly Dance', 
        'MUSIC_PREFERENCE_Bellydance', 'MUSIC_PREFERENCE_Bhangra', 'MUSIC_PREFERENCE_Blues', 
        'MUSIC_PREFERENCE_Bollywood', 'MUSIC_PREFERENCE_Brazilian Rhythms', 'MUSIC_PREFERENCE_Broadway', 
        'MUSIC_PREFERENCE_Caribbean Rhythms', 'MUSIC_PREFERENCE_Chill Out', 'MUSIC_PREFERENCE_Country', 
        'MUSIC_PREFERENCE_Cumbia', 'MUSIC_PREFERENCE_Disco', 'MUSIC_PREFERENCE_Electronic', 
        'MUSIC_PREFERENCE_House', 'MUSIC_PREFERENCE_K-Pop', 'MUSIC_PREFERENCE_Merengue', 'MUSIC_PREFERENCE_Other', 
        'MUSIC_PREFERENCE_Pop', 'MUSIC_PREFERENCE_R&B', 'MUSIC_PREFERENCE_Reggae', 'MUSIC_PREFERENCE_Reggaeton', 
        'MUSIC_PREFERENCE_Rock', 'MUSIC_PREFERENCE_Salsa', 'MUSIC_PREFERENCE_Soca', 'MUSIC_PREFERENCE_Techno', 
        'MUSIC_PREFERENCE_World Rhythms'
    ]
    available_cols = set(churnData.columns)
    variablesModelo = [v for v in variablesModelo if v in available_cols]
    
    xTotal = churnData[variablesModelo]
    yTotal = churnData['PAID_SUBSCRIPTION_CHURN_30D']
    # Se realiza la división de los datos en conjuntos de entrenamiento y prueba.
    xTrain, xTest, yTrain, yTest = train_test_split(xTotal, yTotal, test_size = 0.3, random_state = 0, stratify = yTotal)
    
    # 3. ENTRENAMIENTO DEL MODELO FINAL CON LOS MEJORES PARÁMETROS
    print("--- 3. Inicializando y entrenando el modelo con hiperparámetros pre-optimizados ---")
    
    # Se inicializa el clasificador con los mejores parámetros encontrados previamente.
    finalModel = xgb.XGBClassifier(
        **BEST_PARAMS,
        enable_categorical = True, 
        eval_metric = 'auc', 
        random_state = 0
    )
    
    # Se entrena el modelo en el conjunto de entrenamiento.
    finalModel.fit(xTrain, yTrain) 

    # 4. EVALUACIÓN Y CÁLCULO DE MÉTRICAS FINALES
    print("--- 4. Evaluación de Métricas Finales ---")
    # Se realizan las predicciones en el conjunto de prueba.
    yPred = finalModel.predict(xTest)
    yPredProb = finalModel.predict_proba(xTest)[:, 1]
    
    # Se calculan las métricas de rendimiento.
    modelAUC = roc_auc_score(yTest, yPredProb)
    modelAccuracy = accuracy_score(yTest, yPred)
    modelPrecision = precision_score(yTest, yPred, zero_division=0)
    modelRecall = recall_score(yTest, yPred, zero_division=0)
    modelF1 = f1_score(yTest, yPred, zero_division=0)

    # Se calcula la cantidad de clientes de alto riesgo (Probabilidad de Churn > 0.66).
    highRiskCount = (yPredProb > 0.66).sum()

    # Se obtiene la importancia de las características por ganancia (gain).
    importance = finalModel.get_booster().get_score(importance_type = 'gain')
    importanceDF = pd.DataFrame({'Feature': list(importance.keys()), 'Importance': list(importance.values())}) \
                                 .sort_values(by = 'Importance', ascending = False).head(10)

    # Se genera la distribución de riesgo en tres categorías.
    risk_bins = [0.0, 0.33, 0.66, 1.01]
    risk_labels = ["Bajo Riesgo (0-33%)", "Riesgo Medio (33-66%)", "Alto Riesgo (66-100%)"]
    
    risk_counts = pd.cut(yPredProb, bins=risk_bins, labels=risk_labels, right=False, include_lowest=True).value_counts()
    risk_data = {label: risk_counts.get(label, 0) for label in risk_labels}
    risk_series = pd.Series(risk_data, index=risk_labels)

    print("--- 5. Datos listos para Dash ---")
   
    return {
        "auc": modelAUC, "accuracy": modelAccuracy, "precision": modelPrecision, "recall": modelRecall,
        "f1": modelF1, "highRiskCount": highRiskCount, "testSetSize": xTest.shape[0],
        "importance_df": importanceDF, "risk_counts": risk_series,
    }


# ==============================================================================
# 2. INICIALIZACIÓN DE DASH Y CARGA DE DATOS DINÁMICOS
# ==============================================================================

RESULTS = get_model_results()

# Se asignan resultados a variables globales para el layout.
MODEL_METRICS = {
    "auc": RESULTS["auc"], "accuracy": RESULTS["accuracy"], "precision": RESULTS["precision"],
    "recall": RESULTS["recall"], "f1": RESULTS["f1"], "highRiskCount": RESULTS["highRiskCount"],
    "testSetSize": RESULTS["testSetSize"]
}
IMPORTANCE_DF = RESULTS["importance_df"].sort_values(by='Importance', ascending=True)
RISK_DISTRIBUTION_DATA = RESULTS["risk_counts"]

# Se inicializa la aplicación Dash.
app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Dashboard MLOps - Modelo de Churn Optimizado"
server = app.server
app.config.suppress_callback_exceptions = True

# Estilos de tarjeta base.
CARD_STYLE = {
    'borderRadius': '12px', 'padding': '20px', 'marginBottom': '20px',
    'boxShadow': '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1)',
    'backgroundColor': 'white', 'flexGrow': 1
}

# --- 3. FUNCIONES DE COMPONENTES DASH ---

def kpi_card(title, value, description, color, value_format='{:.3f}', large_text_color='black'):
    """Se genera un componente de tarjeta KPI con formato condicional."""
    
    display_value = ""
    # Se formatea el valor para conteos o porcentajes.
    if 'Count' in description or 'Riesgo' in title or 'Usuarios' in description or 'Total' in title:
        # Formato para conteo de usuarios/total
        display_value = f"{int(value):,}".replace(',', '.')
        
        # Color del texto grande basado en el riesgo (para tarjetas de riesgo)
        if 'Alto Riesgo' in title or (value > 0 and color == HIGH_RISK_RED):
            large_text_color = HIGH_RISK_RED
        elif 'Bajo Riesgo' in title:
            large_text_color = MODEL_BLUE
        elif 'Riesgo Medio' in title:
            large_text_color = '#f59e0b'
        else:
             large_text_color = '#1f2937' # Default para totales
             
    elif value_format == '{:.1%}':
        # Formato de porcentaje
        display_value = f"{value * 100:.1f}%"
        
    else:
        # Formato general de 3 decimales (métricas)
        display_value = value_format.format(value)
        large_text_color = MODEL_BLUE

    if value == 0:
        large_text_color = '#9ca3af'
        
    return html.Div(
        style={**CARD_STYLE, 'borderLeft': f'5px solid {color}'},
        children=[
            html.P(title, style={'fontSize': '14px', 'color': '#6b7280', 'fontWeight': '500'}),
            html.P(display_value, style={'fontSize': '30px', 'fontWeight': 'bold', 'marginTop': '4px', 'color': large_text_color}),
            html.P(description, style={'fontSize': '12px', 'color': '#9ca3af', 'marginTop': '4px'})
        ]
    )

def plot_feature_importance(df):
    """Se genera el gráfico de barras horizontal para la Importancia de Características (Feature Importance)."""
    if df.empty or df['Importance'].sum() == 0:
        return go.Figure().update_layout(title="No se pudo calcular la Importancia de Variables", height=400)
    
    df_plot = df.iloc[::-1].copy()
    
    fig = go.Figure(go.Bar(
        x=df_plot['Importance'], y=df_plot['Feature'], orientation='h',
        marker_color=MODEL_BLUE,
        hovertemplate='<b>%{y}</b><br>Importancia: %{x:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title_text='Top 10 Variables con Mayor Influencia (Gain)',
        height=400, margin={'t': 50, 'b': 20, 'l': 200, 'r': 10},
        plot_bgcolor='white', paper_bgcolor='white',
        xaxis={'title': 'Importancia (Gain)', 'gridcolor': '#e5e7eb', 'showgrid': True},
        yaxis={'title': None, 'tickfont': {'size': 10}},
    )
    return fig

def plot_risk_distribution(counts):
    """Se genera el gráfico de barras para la Distribución de Probabilidad de Riesgo de Churn."""
    if counts.sum() == 0:
        return go.Figure().update_layout(title="No hay datos de distribución de riesgo para mostrar", height=400)
        
    labels = counts.index.tolist()
    values = counts.values.tolist()
    
    colors = [MODEL_BLUE, '#f59e0b', HIGH_RISK_RED] # Ajustado a HIGH_RISK_RED
    
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=colors,
        hovertemplate='<b>%{x}</b><br>Usuarios: %{y:,}<extra></extra>'
    ))

    fig.update_layout(
        title_text='Distribución de Probabilidad de Churn (Test Set)',
        height=400,
        margin={'t': 50, 'b': 50, 'l': 50, 'r': 50},
        plot_bgcolor='white', paper_bgcolor='white',
        xaxis={'title': None, 'categoryorder':'array', 'categoryarray': ["Bajo Riesgo (0-33%)", "Riesgo Medio (33-66%)", "Alto Riesgo (66-100%)"]},
        yaxis={'title': 'Usuarios (Conteo)', 'gridcolor': '#e5e7eb', 'tickformat': ','},
    )
    return fig

# --- 4. LAYOUT DE LA APLICACIÓN DASH ---

# Se define el encabezado principal del dashboard.
header = html.Header(
    style={'marginBottom': '30px', 'borderBottom': '1px solid #e5e7eb', 'paddingBottom': '20px'},
    children=[
        html.H1(
            "Sistema de Prevención del Churn - Modelo XGBoost",
            style={'fontSize': '28px', 'fontWeight': '800', 'color': '#1f2937'}
        ),
        html.P(
            f"Resultados del Modelo XGBoost Optimizado (AUC={MODEL_METRICS['auc']:.3f}) evaluado sobre el Test Set. Total de usuarios en Test Set: {MODEL_METRICS['testSetSize']:,}.",
            style={'color': '#6b7280', 'marginTop': '4px'}
        )
    ]
)

# Se organiza la sección de Métricas de Negocio.
kpis_business = html.Div(
    className='row',
    style={'display': 'flex', 'gap': '20px', 'marginBottom': '30px'},
    children=[
        kpi_card(
            title="Total de Usuarios Evaluados", value=MODEL_METRICS["testSetSize"],
            description="Tamaño del conjunto de datos de prueba.", color=PURPLE, 
        ),
    ]
)

# Desglose por Segmentos de Riesgo (Bajo, Medio, Alto)
risk_breakdown = html.Div(
    className='row',
    style={'display': 'flex', 'gap': '20px', 'marginBottom': '30px', 'marginTop': '10px'},
    children=[
        kpi_card(
            title="Bajo Riesgo (0-33%)", value=RISK_DISTRIBUTION_DATA.get("Bajo Riesgo (0-33%)", 0),
            description="Clientes estables con baja probabilidad de Churn (Conservar).", color=MODEL_BLUE,
        ),
        kpi_card(
            title="Riesgo Medio (33-66%)", value=RISK_DISTRIBUTION_DATA.get("Riesgo Medio (33-66%)", 0),
            description="Clientes a monitorear; potencial para ofertas de engagement (Monitorear).", color='#f59e0b',
        ),
        kpi_card(
            title="Alto Riesgo (66-100%)", value=RISK_DISTRIBUTION_DATA.get("Alto Riesgo (66-100%)", 0),
            description="Clientes en riesgo inminente de rotación (Prioridad de Retención).", color=HIGH_RISK_RED,
        ),
    ]
)


# Se organiza la sección de Métricas de Calidad del Modelo.
kpis_model = html.Div(
    className='row',
    style={'display': 'flex', 'gap': '10px', 'marginBottom': '30px'},
    children=[
        kpi_card("ROC AUC", MODEL_METRICS["auc"], "Capacidad predictiva del modelo optimizado.", MODEL_BLUE),
        kpi_card("Accuracy", MODEL_METRICS["accuracy"], "Tasa general de predicciones correctas.", MODEL_BLUE, value_format='{:.1%}'),
        kpi_card("Precision", MODEL_METRICS["precision"], "Exactitud en las predicciones positivas.", MODEL_BLUE, value_format='{:.1%}'),
        kpi_card("Recall", MODEL_METRICS["recall"], "Capacidad de capturar Churners reales.", MODEL_BLUE, value_format='{:.1%}'),
        kpi_card("F1 Score", MODEL_METRICS["f1"], "Métrica de equilibrio (Precision y Recall).", MODEL_BLUE),
    ]
)

# Se organiza la sección de Gráficos (Importancia y Distribución).
# MODIFICACIÓN CLAVE: Se usa 'flex: 1' y 'minWidth' para asegurar que ocupen la mitad del espacio en pantallas grandes.
charts_section = html.Div(
    className='row',
    style={'display': 'flex', 'gap': '20px', 'marginBottom': '30px', 'flexWrap': 'wrap'},
    children=[
        html.Div(
            # Configurado para tomar el 50% del ancho con flex: 1, pero se envuelve en pantallas estrechas (<800px)
            style={**CARD_STYLE, 'flex': 1, 'minWidth': '400px'},
            children=[dcc.Graph(figure=plot_feature_importance(IMPORTANCE_DF))]
        ),
        html.Div(
            # Configurado para tomar el 50% del ancho con flex: 1, pero se envuelve en pantallas estrechas (<800px)
            style={**CARD_STYLE, 'flex': 1, 'minWidth': '400px'},
            children=[dcc.Graph(figure=plot_risk_distribution(RISK_DISTRIBUTION_DATA))]
        )
    ]
)

# Se define el layout final de la aplicación.
app.layout = html.Div(
    id="app-container",
    style={'padding': '20px', 'maxWidth': '1400px', 'margin': '0 auto', 'backgroundColor': '#f7f9fc', 'fontFamily': 'Inter, sans-serif'},
    children=[
        header,
        html.H2("1. Total de Usuarios y Segmentación de Riesgo", style={'fontSize': '20px', 'fontWeight': 'bold', 'marginBottom': '10px'}),
        kpis_business,
        html.H3("Desglose de Clientes por Segmento de Riesgo", style={'fontSize': '18px', 'fontWeight': '600', 'marginBottom': '10px', 'marginTop': '20px'}),
        risk_breakdown,
        html.H2("2. Calidad y Estabilidad del Modelo Optimizado", style={'fontSize': '20px', 'fontWeight': 'bold', 'marginBottom': '10px', 'marginTop': '30px'}),
        kpis_model,
        html.H2("3. Insights y Distribución de Probabilidad", style={'fontSize': '20px', 'fontWeight': 'bold', 'marginBottom': '10px'}),
        charts_section,
    ],
)

if __name__ == "__main__": app.run(debug=True)
