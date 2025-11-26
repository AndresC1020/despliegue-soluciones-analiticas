# Zumba App | Manual de Usuario del Tablero de Churn

## 1. Objetivo del tablero

El tablero tiene como propósito apoyar a los equipos de Producto y Growth Marketing de Zumba
en la identificación temprana de usuarios con alta probabilidad de abandono y en el
diseño de campañas de retención personalizadas.

Las visualizaciones se construyen a partir de un archivo de predicciones generado por el
modelo de churn (`predictions_ChurnPredictedData.csv`), donde cada fila representa un usuario
con su probabilidad estimada de abandono.

---

## 2. Fuentes de información

El tablero utiliza como fuente principal el archivo:

- `data/predictions_ChurnPredictedData.csv`

Este archivo contiene las columnas:

- `PREDICTED_PROBABILITY`: probabilidad estimada de churn (0 a 1).
- `PREDICTED_CLASS`: etiqueta binaria (1 = churn, 0 = no churn).
- `COUNTRY`: país del usuario.
- Variables de comportamiento y preferencias utilizadas para el modelo.

El tablero **no recalcula el modelo**, sino que consume las predicciones ya generadas.

---

## 3. Descripción de las secciones del tablero

### 3.1. Encabezado

- **Sistema de Prevención del Churn**

Texto descriptivo indicando que los resultados se basan en el archivo de
generadas, junto con el **total de usuarios evaluados**.

---

### 3.2. Módulo 1 – Total de Usuarios y Segmentación de Riesgo

Este módulo muestra:

1. **Total de Usuarios Evaluados**  
   - Número total de registros presentes en el archivo de predicciones.
   - Permite dimensionar el tamaño de la base sobre la cual se analiza el churn.

2. **Desglose de Clientes por Segmento de Riesgo**  
   Los usuarios se agrupan en tres segmentos, de acuerdo con su probabilidad de churn:

   - **Bajo Riesgo (0–33%)**  
     Usuarios estables, con probabilidad baja de cancelación. No requieren acciones intensivas.

   - **Riesgo Medio (33–66%)**  
     Usuarios con probabilidad intermedia, ideales para campañas de engagement de bajo costo
     (notificaciones, recomendaciones de contenido, recordatorios de metas).

   - **Alto Riesgo (66–100%)**  
     Usuarios con alta probabilidad de rotación. Son el foco principal de las campañas de retención
     personalizadas (descuentos, contacto directo, beneficios especiales).

Cada tarjeta presenta el número de usuarios en cada segmento.

---

### 3.3. Módulo 2 – Calidad y Estabilidad del Modelo

Este módulo muestra las métricas de desempeño del modelo utilizado para generar las predicciones:

- **ROC AUC**: capacidad del modelo para discriminar entre usuarios que abandonan y los que se quedan.
- **Accuracy**: porcentaje global de aciertos.
- **Precision**: proporción de usuarios marcados como “churn” que efectivamente abandonan.
- **Recall**: proporción de usuarios que abandonan y el modelo logra identificar (métrica clave en churn).
- **F1 Score**: equilibrio entre Precision y Recall.

Estas métricas permiten evaluar si el modelo es lo suficientemente confiable para soportar decisiones de negocio.

---

### 3.4. Módulo 3 – Insights y Distribución de Probabilidad

Este módulo contiene dos gráficos principales:

1. **Top 10 Variables con Mayor Influencia**  
   - Muestra las variables con mayor importancia relativa en el modelo (basadas en la varianza de las features).
   - Ayuda a entender qué factores están más asociados al churn (por ejemplo, porcentaje de video visto, número de clases, etc.).

2. **Distribución de Probabilidad de Churn**  
   - Muestra cuántos usuarios se concentran en cada segmento de riesgo.
   - Permite validar si las predicciones están balanceadas o concentradas en un rango específico.

---

### 3.5. Módulo 4 – Análisis por País

Este módulo permite analizar la probabilidad de churn a nivel geográfico.

1. **Selector de país**  
   - El usuario puede elegir un país en la lista desplegable.
   - El gráfico “Distribución de Riesgo por País” muestra el número de usuarios de ese país en cada segmento de riesgo (Bajo, Medio, Alto).

2. **Mapa de calor por segmento de riesgo**  
   - El usuario selecciona un segmento de riesgo (Bajo, Medio o Alto).
   - El mapa de calor muestra el porcentaje de usuarios de cada país que pertenecen a ese segmento.
   - Este gráfico permite identificar regiones donde el churn es especialmente alto o bajo para cada nivel de riesgo.

---

## 4. Casos de uso sugeridos

- **Diseño de campañas de retención:**  
  Utilizar el segmento de “Alto Riesgo” como objetivo principal para campañas más intensivas.

- **Optimización de contenidos:**  
  Revisar las variables más influyentes en el churn para proponer ajustes en el catálogo de clases
  (por ejemplo, mejorar la oferta de cierto tipo de clases o música).

- **Enfoque geográfico:**  
  Utilizar el análisis por país para priorizar mercados con mayor nivel de riesgo.

---

## 5. Limitaciones actuales

- El tablero depende de un archivo de predicciones pre-generadas; no recalcula el modelo en tiempo real.
- Las métricas de modelo son fijas y corresponden a una versión específica del modelo entrenado.
- Cualquier cambio en la estructura del archivo CSV requiere ajustar el código antes de volver a ejecutar el tablero.

---

## 6. Contacto y soporte

Para dudas técnicas o solicitudes de mejora sobre el tablero, contactar al equipo de analítica
responsable del proyecto de churn de Zumba.