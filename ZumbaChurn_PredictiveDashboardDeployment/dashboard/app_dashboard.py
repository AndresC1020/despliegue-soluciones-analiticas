# ============================================================
# DASHBOARD FINAL DE CHURN BASADO SOLO EN PREDICCIONES
# Incluye Métricas del Modelo, Importancia de Variables y Tabla de Detalle.
# Fuente de datos: predictions_ChurnPredictedData.csv
# OPTIMIZADO PARA EJECUCIÓN LOCAL
# ============================================================

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import dash_table 
import os

# ============================================================
# 1. CARGA DEL ARCHIVO DE PREDICCIONES (Robusta y Flexible)
# ============================================================

# Definiciones de constantes de diseño
MODEL_BLUE = '#00c7e5'
HIGH_RISK_RED = '#ef4444'
MEDIUM_RISK_ORANGE = '#f59e0b'
LOW_RISK_BLUE = '#3b82f6'
FILENAME = 'predictions_ChurnPredictedData.csv'

# --- MANEJO ROBUSTO DE RUTAS PARA EJECUCIÓN LOCAL ---
df = pd.DataFrame() # Inicializar como vacío

# Estrategia de carga robusta: 
# 1. Buscar en el directorio actual (ejecución directa)
# 2. Buscar en ./data/ (estructura de carpetas)
try:
    if os.path.exists(FILENAME):
        data_path = FILENAME
    elif os.path.exists(os.path.join('data', FILENAME)):
        data_path = os.path.join('data', FILENAME)
    else:
        raise FileNotFoundError(f"No se encontró el archivo {FILENAME} en el directorio actual ni en ./data/")

    df = pd.read_csv(data_path)
    
    # Eliminar primera columna vacía si existe (basado en la plantilla)
    if not df.empty and (df.columns[0].startswith("Unnamed") or df.columns[0] == ""):
        df = df.drop(df.columns[0], axis=1)

    # --- MEJORA DE ROBUSTEZ: Normalización de nombres de columnas ---
    rename_map = {
        'predicted_probability': 'PREDICTED_PROBABILITY',
        'predicted_class': 'PREDICTED_CLASS',
        'country': 'COUNTRY',
        'subscription_tier': 'SUBSCRIPTION_TIER',
        'subscription_provider': 'SUBSCRIPTION_PROVIDER',
        'trial_duration_days': 'TRIAL_DURATION_DAYS',
        'no_video_watched_30d': 'NO_VIDEO_WATCHED_30D', 
    }
    # Buscar las claves que existen en el DataFrame (ignorando mayúsculas/minúsculas)
    lower_col_map = {col.lower(): col for col in df.columns}
    final_rename_map = {}
    for key, val in rename_map.items():
        if key.lower() in lower_col_map:
            final_rename_map[lower_col_map[key.lower()]] = val
        
    df = df.rename(columns=final_rename_map)
    # -------------------------------------------------------------

except FileNotFoundError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"Error al leer el CSV: {e}")

# Manejo de DataFrame vacío o columnas necesarias faltantes
if not df.empty and 'PREDICTED_PROBABILITY' in df.columns:
    
    # Rellenar valores nulos en la columna COUNTRY si existe, o crearla como 'Unknown'
    if 'COUNTRY' in df.columns:
        df['COUNTRY'] = df['COUNTRY'].fillna('Unknown')
    else:
        df['COUNTRY'] = 'Unknown'

    # 1. Crear ID de Usuario a partir del índice (CRUCIAL para la tabla)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'USER_ID'}, inplace=True)
    df['USER_ID'] = df['USER_ID'].astype(str) # Convertir a string para búsqueda en la tabla

    # ============================================================
    # 2. CREAR SEGMENTO DE RIESGO
    # ============================================================

    df["RISK_SEGMENT"] = pd.cut(
        df["PREDICTED_PROBABILITY"],
        bins=[0, 0.33, 0.66, 1.0],
        labels=["Bajo Riesgo (0-33%)", "Riesgo Medio (33-66%)", "Alto Riesgo (66-100%)"],
        include_lowest=True
    )

    risk_counts = df["RISK_SEGMENT"].value_counts().reindex(
        ["Bajo Riesgo (0-33%)", "Riesgo Medio (33-66%)", "Alto Riesgo (66-100%)"]
    ).fillna(0)
    TOTAL_USERS = len(df)

    # ============================================================
    # 3. IMPORTANCIA DE VARIABLES (VARIANZA)
    # ============================================================

    # Excluir columnas de predicción y categóricas clave del cálculo de importancia
    exclude_cols = ["USER_ID", "PREDICTED_PROBABILITY", "PREDICTED_CLASS", "RISK_SEGMENT", "COUNTRY", "SUBSCRIPTION_TIER", "SUBSCRIPTION_PROVIDER", "TRIAL_DURATION_DAYS", "NO_VIDEO_WATCHED_30D"]
    
    feature_columns = [
        col for col in df.columns
        if col not in exclude_cols
    ]

    # Calcular varianza solo en columnas numéricas
    numeric_df = df[feature_columns].select_dtypes(include=[np.number]).dropna(axis=1, how='all')
    
    # Si quedan columnas numéricas después de la limpieza:
    if not numeric_df.empty and len(numeric_df.columns) > 1:
        # Llenar NaN con 0 antes de calcular la varianza (para mayor robustez)
        numeric_df = numeric_df.fillna(0) 
        importance = numeric_df.var().sort_values(ascending=False).head(10)
        importance_df = pd.DataFrame({"Feature": importance.index, "Importance": importance.values})
    else:
        importance_df = pd.DataFrame({"Feature": ["No hay datos numéricos para calcular Varianza"], "Importance": [0]})

    
    # Opciones de País para el Dropdown
    country_list = df["COUNTRY"].unique()
    default_country = "United States of America" if "United States of America" in country_list else (sorted(country_list)[0] if country_list.size > 0 else "Unknown")
    
    # Datos iniciales para la tabla (las primeras 100 filas)
    display_cols = ["USER_ID", "PREDICTED_PROBABILITY", "RISK_SEGMENT", "COUNTRY", "SUBSCRIPTION_TIER", "TRIAL_DURATION_DAYS", "NO_VIDEO_WATCHED_30D"]
    initial_table_data = df[display_cols].head(100).to_dict('records')


else:
    # Caso de DataFrame vacío (manejo de errores)
    print("El DataFrame está vacío o faltan columnas clave. El dashboard mostrará valores predeterminados (0).")
    risk_counts = pd.Series([0, 0, 0], index=["Bajo Riesgo (0-33%)", "Riesgo Medio (33-66%)", "Alto Riesgo (66-100%)"])
    TOTAL_USERS = 0
    importance_df = pd.DataFrame({"Feature": ["Error de Carga o Datos"], "Importance": [0]})
    country_list = ["Error"]
    default_country = "Error"
    initial_table_data = []

    
# ============================================================
# 4. MÉTRICAS DEL MODELO (Hardcoded - Asumidas de un entrenamiento)
# ============================================================

MODEL_METRICS = {
    "auc": 0.659,
    "accuracy": 0.614,
    "precision": 0.584,
    "recall": 0.528,
    "f1": 0.554
}

# ============================================================
# 5. ESTILOS Y COMPONENTES (KPI Cards y Configuración de Gráficas)
# ============================================================

CARD_STYLE = {
    'borderRadius': '12px',
    'padding': '20px',
    'backgroundColor': 'white',
    'boxShadow': '0 3px 10px rgba(0,0,0,0.12)',
    'flex': '1',
    'minWidth': '200px'
}

# Configuración global para los gráficos de Plotly (desactiva el zoom)
GRAPH_CONFIG = {
    'scrollZoom': False, 
    'displayModeBar': False 
}

def kpi_card(title, value, description, color=MODEL_BLUE):
    """Genera un componente de tarjeta KPI."""
    
    # Formateo condicional para métricas o porcentajes
    if isinstance(value, float) and value < 1.0:
        display_value = f"{value:.3f}"
    elif isinstance(value, (int, np.integer)):
        # Usamos f"{value:,}" para el separador de miles y luego reemplazamos la coma por el punto
        display_value = f"{value:,}".replace(',', '.') 
    else:
        display_value = str(value)

    # Convertimos a int para evitar problemas de tipos si viene de Series
    if isinstance(value, np.integer):
        value = int(value)
        
    return html.Div(
        # Añadir barra de color a la izquierda
        style={**CARD_STYLE, 'borderLeft': f'5px solid {color}'},
        children=[
            html.P(title, style={'color':'#6b7280','fontSize':'14px','marginBottom':'4px'}),
            html.H3(
                display_value,
                style={'color':color,'fontWeight':'700','fontSize':'28px','margin':'0'}
            ),
            html.P(description, style={'color':'#9ca3af','fontSize':'12px','marginTop':'6px'})
        ]
    )

# ============================================================
# 6. GRÁFICAS
# ============================================================

def plot_feature_importance(df_importance):
    """Gráfico de barras de la importancia de variables (basado en Varianza)."""
    fig = go.Figure(go.Bar(
        x=df_importance["Importance"][::-1],
        y=df_importance["Feature"][::-1],
        orientation="h",
        marker_color=MODEL_BLUE
    ))
    fig.update_layout(
        title="Top 10 Variables con Mayor Varianza (Influencia)",
        height=380,
        margin=dict(l=160, r=20, t=50, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis={'title': 'Varianza Normalizada'},
        title_font_size=14
    )
    return fig


def plot_risk_distribution():
    """Gráfico de distribución global de los segmentos de riesgo."""
    
    labels = risk_counts.index
    values = risk_counts.values
    colors = [LOW_RISK_BLUE, MEDIUM_RISK_ORANGE, HIGH_RISK_RED]

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[f"{v:,}".replace(',', '.') for v in values],
        textposition="outside",
        hovertemplate='<b>%{x}</b><br>Usuarios: %{y:,}<extra></extra>'
    ))
    fig.update_layout(
        title="Distribución Global de Probabilidad de Churn",
        yaxis_title="Usuarios (Conteo)",
        height=380,
        plot_bgcolor="white",
        paper_bgcolor="white",
        yaxis=dict(gridcolor="#e5e7eb"),
        title_font_size=14
    )
    return fig


def plot_risk_by_country(country):
    """Gráfico de distribución de riesgo filtrado por país."""
    if df.empty or 'COUNTRY' not in df.columns:
        return go.Figure().update_layout(title="Datos no disponibles")
        
    sub = df[df["COUNTRY"].astype(str) == str(country)]
    title_suffix = f" en {country}"
    
    if sub.empty:
        return go.Figure().update_layout(title=f"No hay datos de predicción para {country}")

    counts = sub["RISK_SEGMENT"].value_counts().reindex(
        ["Bajo Riesgo (0-33%)","Riesgo Medio (33-66%)","Alto Riesgo (66-100%)"]
    ).fillna(0)

    fig = go.Figure(go.Bar(
        x=counts.index,
        y=counts.values,
        marker_color=[LOW_RISK_BLUE, MEDIUM_RISK_ORANGE, HIGH_RISK_RED],
        text=[f"{v:,}".replace(',', '.') for v in counts.values],
        textposition="outside",
        hovertemplate='<b>%{x}</b><br>Usuarios: %{y:,}<extra></extra>'
    ))

    fig.update_layout(
        title=f"Distribución de Riesgo{title_suffix}",
        height=380,
        margin=dict(l=20,r=20,t=50,b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        yaxis=dict(title="Usuarios", gridcolor="#e5e7eb"),
        title_font_size=14
    )

    return fig


def plot_country_heatmap(segment):
    """Mapa de Calor de Concentración de Riesgo por País."""
    if df.empty or 'COUNTRY' not in df.columns:
        return go.Figure().update_layout(title="Datos no disponibles")
        
    ranges = {
        "Bajo Riesgo (0-33%)": (0,0.33),
        "Riesgo Medio (33-66%)": (0.33,0.66),
        "Alto Riesgo (66-100%)": (0.66,1.0)
    }

    low, high = ranges[segment]

    # Calcular qué porcentaje de los usuarios de cada país están en el segmento
    df["IN_SEGMENT"] = df["PREDICTED_PROBABILITY"].between(low, high)

    summary = df.groupby("COUNTRY")["IN_SEGMENT"].mean().reset_index()
    summary["Pct"] = summary["IN_SEGMENT"] * 100

    # Establecer rango de color dinámico
    summary_max_pct = summary["Pct"].max() if not summary.empty else 10
    color_max = max(5, summary_max_pct * 1.05) 

    color_scale = "Blues" if segment == "Bajo Riesgo (0-33%)" else ("Reds" if segment == "Alto Riesgo (66-100%)" else "Oranges")

    fig = px.choropleth(
        summary,
        locations="COUNTRY",
        locationmode="country names",
        color="Pct",
        color_continuous_scale=color_scale,
        range_color=[0, color_max],
        title=f"Concentración de Usuarios en Segmento: {segment}"
    )

    fig.update_layout(
        height=400,
        margin=dict(l=20,r=20,t=50,b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        geo=dict(
            bgcolor="rgba(0,0,0,0)",
            showframe=False,
            showcoastlines=True,
            coastlinecolor="#ccc",
            projection_scale=1
        ),
        title_font_size=14
    )

    fig.update_coloraxes(colorbar=dict(
        title="% usuarios",
        thickness=10,
        len=0.7
    ))

    return fig


# ============================================================
# 7. CREAR APP
# ============================================================

# Tailwind-like CDN para una mejor tipografía global (Inter)
external_css = [
    "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap"
]

app = dash.Dash(
    __name__, 
    external_stylesheets=external_css,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server

app.title = "Dashboard Churn – Predicciones MLOps"

# ============================================================
# 8. LAYOUT
# ============================================================

app.layout = html.Div(style={
    'padding':'20px',
    'maxWidth':'1400px',
    'margin':'0 auto',
    'backgroundColor':'#f7f9fc',
    'fontFamily':'Inter, sans-serif'
}, children=[

    html.H2("Sistema de Prevención del Churn - MLOps",
            style={'fontWeight':'800','fontSize':'26px','color':'#1f2937'}),

    html.P(
        f"Resultados basados en el archivo de predicciones pre-generadas. "
        f"Total de usuarios evaluados: {TOTAL_USERS:,}.".replace(',', '.'),
        style={'color':'#6b7280','marginBottom':'25px'}
    ),

    # ------------------------------------------------------------
    # 1. TOTAL USUARIOS & RIESGO
    # ------------------------------------------------------------
    html.H3("1. Segmentación de Riesgo Global", style={'fontSize':'20px','fontWeight':'700', 'borderBottom': '1px solid #e5e7eb', 'paddingBottom': '10px'}),

    html.Div(style={
        'display':'flex',
        'gap':'20px',
        'flexWrap':'wrap',
        'marginBottom':'40px'
    }, children=[
        kpi_card("Total de Usuarios Evaluados", TOTAL_USERS,
                 "Tamaño total del archivo de predicciones.", color='#1f2937'),
        kpi_card("Bajo Riesgo (0-33%)", risk_counts.get("Bajo Riesgo (0-33%)",0),
                 "Clientes estables con baja probabilidad de Churn.", LOW_RISK_BLUE),
        kpi_card("Riesgo Medio (33-66%)", risk_counts.get("Riesgo Medio (33-66%)",0),
                 "Clientes a monitorear; potencial para ofertas.", MEDIUM_RISK_ORANGE),
        kpi_card("Alto Riesgo (66-100%)", risk_counts.get("Alto Riesgo (66-100%)",0),
                 "Clientes en riesgo inminente de rotación.", HIGH_RISK_RED)
    ]),

    # ------------------------------------------------------------
    # 2. MÉTRICAS FIJAS DEL MODELO
    # ------------------------------------------------------------
    html.H3("2. Métricas de Calidad del Modelo", style={'fontSize':'20px','fontWeight':'700', 'borderBottom': '1px solid #e5e7eb', 'paddingBottom': '10px'}),
    
    html.Div(style={
        'display':'flex',
        'gap':'20px',
        'flexWrap':'wrap',
        'justifyContent':'space-between',
        'marginBottom':'40px'
    }, children=[
        kpi_card("ROC AUC", MODEL_METRICS["auc"], "Capacidad predictiva.", MODEL_BLUE),
        kpi_card("Accuracy", MODEL_METRICS["accuracy"], "Tasa de aciertos.", MODEL_BLUE),
        kpi_card("Precision", MODEL_METRICS["precision"], "Exactitud de positivos.", MODEL_BLUE),
        kpi_card("Recall", MODEL_METRICS["recall"], "Capacidad de capturar Churn.", MODEL_BLUE),
        kpi_card("F1 Score", MODEL_METRICS["f1"], "Balance entre Precision y Recall.", MODEL_BLUE)
    ]),

    # ------------------------------------------------------------
    # 3. INSIGHTS Y DISTRIBUCIÓN
    # ------------------------------------------------------------
    html.H3("3. Insights Globales y Distribución de Probabilidad", style={'fontSize':'20px','fontWeight':'700', 'borderBottom': '1px solid #e5e7eb', 'paddingBottom': '10px'}),

    html.Div(style={
        'display':'flex',
        'gap':'20px',
        'marginBottom':'40px',
        'flexWrap': 'wrap'
    }, children=[
        # Aplicamos la configuración para desactivar scrollZoom y limpiar la barra de herramientas
        html.Div(style={**CARD_STYLE, 'flex':'1'}, children=[dcc.Graph(figure=plot_feature_importance(importance_df), config=GRAPH_CONFIG)]),
        html.Div(style={**CARD_STYLE, 'flex':'1'}, children=[dcc.Graph(figure=plot_risk_distribution(), config=GRAPH_CONFIG)])
    ]),

    # ------------------------------------------------------------
    # 4. ANÁLISIS INTERACTIVO POR PAÍS
    # ------------------------------------------------------------
    html.H3("4. Análisis Interactivo por Geografía", style={'fontSize':'20px','fontWeight':'700', 'borderBottom': '1px solid #e5e7eb', 'paddingBottom': '10px'}),

    html.Div(style={
        'display':'flex',
        'gap':'20px',
        'marginBottom':'40px',
        'flexWrap':'wrap'
    }, children=[

        # Gráfico de Riesgo por País (Filtro 1)
        html.Div(style={**CARD_STYLE, 'flex':'1', 'minWidth': '400px'}, children=[
            html.Label("Seleccione un país para ver el desglose de riesgo:", style={'fontWeight':'600', 'marginBottom':'5px'}),
            dcc.Dropdown(
                id="country_selector",
                options=[{"label":c, "value":c} for c in sorted(country_list)],
                value=default_country,
                clearable=False,
                style={'width':'100%','marginBottom':'15px', 'maxWidth':'400px'}
            ),
            # Aplicamos la configuración
            dcc.Graph(id="risk_by_country_chart", config=GRAPH_CONFIG)
        ]),

        # Mapa de Calor por Segmento de Riesgo (Filtro 2)
        html.Div(style={**CARD_STYLE, 'flex':'1', 'minWidth': '400px'}, children=[
            html.Label("Seleccione el Segmento de Riesgo para el Mapa Global:", style={'fontWeight':'600', 'marginBottom':'5px'}),
            dcc.Dropdown(
                id="risk_selector",
                options=[
                    {"label":"Alto Riesgo (66-100%)","value":"Alto Riesgo (66-100%)"},
                    {"label":"Riesgo Medio (33-66%)","value":"Riesgo Medio (33-66%)"},
                    {"label":"Bajo Riesgo (0-33%)","value":"Bajo Riesgo (0-33%)"}
                ],
                value="Alto Riesgo (66-100%)", # Enfocamos en el más crítico por defecto
                clearable=False,
                style={'width':'100%','marginBottom':'15px', 'maxWidth':'400px'}
            ),
            # Aplicamos la configuración
            dcc.Graph(id="heatmap_chart", config=GRAPH_CONFIG)
        ])
    ]),
    
    # ------------------------------------------------------------
    # 5. TABLA DE DATOS DETALLADA Y FILTROS (NUEVO)
    # ------------------------------------------------------------
    html.H3("5. Detalle de Usuarios y Filtrado", style={'fontSize':'20px','fontWeight':'700', 'borderBottom': '1px solid #e5e7eb', 'paddingBottom': '10px', 'marginTop': '10px'}),

    html.Div(style={
        'display': 'flex',
        'gap': '20px',
        'marginBottom': '20px',
        'flexWrap': 'wrap'
    }, children=[
        html.Div([
            html.Label("Filtrar por ID de Usuario (Contiene):", style={'fontWeight':'600', 'marginBottom':'5px', 'display': 'block'}),
            dcc.Input(
                id='user_id_input',
                type='text',
                placeholder='Escriba el ID de usuario...',
                debounce=True, # Espera a que el usuario termine de escribir
                style={'padding': '10px', 'borderRadius': '6px', 'border': '1px solid #ccc', 'width': '250px'}
            )
        ]),
        html.Div([
            html.Label("Filtrar por Segmento de Riesgo:", style={'fontWeight':'600', 'marginBottom':'5px', 'display': 'block'}),
            dcc.Dropdown(
                id='risk_segment_filter',
                options=[
                    {'label': 'Todos', 'value': 'ALL'},
                    {"label":"Alto Riesgo (66-100%)","value":"Alto Riesgo (66-100%)"},
                    {"label":"Riesgo Medio (33-66%)","value":"Riesgo Medio (33-66%)"},
                    {"label":"Bajo Riesgo (0-33%)","value":"Bajo Riesgo (0-33%)"}
                ],
                value='ALL',
                clearable=False,
                style={'width': '250px'}
            )
        ])
    ]),

    html.Div(style={**CARD_STYLE, 'padding': '10px 20px', 'overflowX': 'auto'}, children=[
        dash_table.DataTable(
            id='user_detail_table',
            columns=[
                {"name": "ID", "id": "USER_ID"},
                {"name": "Prob. Churn", "id": "PREDICTED_PROBABILITY", "type": "numeric", "format": dash_table.Format.Format(precision=4, scheme=dash_table.Format.Scheme.fixed)},
                {"name": "Riesgo", "id": "RISK_SEGMENT"},
                {"name": "País", "id": "COUNTRY"},
                {"name": "Suscripción", "id": "SUBSCRIPTION_TIER"},
                {"name": "Duración Trial (Días)", "id": "TRIAL_DURATION_DAYS"},
                {"name": "Videos Vistos (30D)", "id": "NO_VIDEO_WATCHED_30D"}
            ],
            data=initial_table_data, 
            style_header={
                'backgroundColor': MODEL_BLUE,
                'color': 'white',
                'fontWeight': 'bold',
                'textAlign': 'left'
            },
            style_data_conditional=[
                {
                    'if': {'column_id': 'RISK_SEGMENT', 'filter_query': '{RISK_SEGMENT} eq "Alto Riesgo (66-100%)"'},
                    'backgroundColor': '#fee2e2', 'color': HIGH_RISK_RED
                },
                {
                    'if': {'column_id': 'RISK_SEGMENT', 'filter_query': '{RISK_SEGMENT} eq "Riesgo Medio (33-66%)"'},
                    'backgroundColor': '#fff5d5', 'color': MEDIUM_RISK_ORANGE
                },
            ],
            style_cell={'textAlign': 'left', 'padding': '10px', 'fontFamily': 'Inter, sans-serif'},
            page_action='native', # Usar paginación nativa
            page_current=0,
            page_size=10, # 10 filas por página
            sort_action="native",
            style_table={'overflowY': 'auto', 'maxHeight': '500px'} # Contenedor con scroll
        )
    ])
])

# ============================================================
# 9. CALLBACKS
# ============================================================

@app.callback(
    Output("risk_by_country_chart", "figure"),
    Input("country_selector", "value")
)
def update_country_chart(country):
    """Actualiza el gráfico de barras de riesgo por país."""
    return plot_risk_by_country(country)


@app.callback(
    Output("heatmap_chart", "figure"),
    Input("risk_selector", "value")
)
def update_heatmap(segment):
    """Actualiza el mapa de calor (Choropleth) de concentración de riesgo."""
    return plot_country_heatmap(segment)

@app.callback(
    Output('user_detail_table', 'data'),
    [
        Input('user_id_input', 'value'),
        Input('risk_segment_filter', 'value')
    ]
)
def update_table_data(user_id_search, risk_segment_value):
    """
    Filtra los datos de la tabla de detalles de usuario según el ID de usuario 
    y el segmento de riesgo seleccionado.
    """
    global df
    if df.empty:
        return []

    filtered_df = df.copy()

    # 1. Filtrar por Segmento de Riesgo
    if risk_segment_value and risk_segment_value != 'ALL':
        filtered_df = filtered_df[filtered_df['RISK_SEGMENT'] == risk_segment_value]

    # 2. Filtrar por ID de Usuario (búsqueda parcial o contiene)
    if user_id_search:
        # Busca cualquier USER_ID que contenga el texto (más flexible)
        search_lower = str(user_id_search).strip().lower()
        # Aseguramos que la columna USER_ID sea string antes de usar str.contains
        try:
            filtered_df = filtered_df[filtered_df['USER_ID'].astype(str).str.contains(search_lower, case=False, na=False)]
        except Exception:
            # Si hay un error, devolvemos el DataFrame sin filtrar por ID
            pass

    # Seleccionar solo las columnas a mostrar
    display_cols = ["USER_ID", "PREDICTED_PROBABILITY", "RISK_SEGMENT", "COUNTRY", "SUBSCRIPTION_TIER", "TRIAL_DURATION_DAYS", "NO_VIDEO_WATCHED_30D"]
    
    # Devolver el DataFrame filtrado como lista de diccionarios
    return filtered_df[display_cols].to_dict('records')

# ============================================================
# 10. EJECUTAR APP
# ============================================================
if __name__ == "__main__":
    print("Dashboard disponible en http://localhost:8050")
    # Nota: Si estás usando gunicorn para el despliegue, el puerto 8050 es el predeterminado.
    # Para ejecución local con dash:
    app.run(debug=True)
