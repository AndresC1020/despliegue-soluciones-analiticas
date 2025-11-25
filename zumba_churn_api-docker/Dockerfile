FROM python:3.12-slim

# Crear usuario no-root
RUN adduser --disabled-password --gecos '' api-user
WORKDIR /opt/zumba-churn-api

# --- 1. INSTALACIÓN DE DEPENDENCIAS ---
# Copiar el archivo de requisitos de la API (Origen: zumba-churn-api/requirements.txt)
COPY zumba-churn-api/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install -r requirements.txt

# --- 2. INSTALACIÓN DEL PAQUETE DEL MODELO (.WHL) ---
# Copiar el paquete WHL (Origen: zumba-churn-api/model-pkg/zumba_churn_pkg-0.0.1-py3-none-any.whl)
COPY zumba-churn-api/model-pkg/zumba_churn_pkg-0.0.1-py3-none-any.whl .
RUN pip install zumba_churn_pkg-0.0.1-py3-none-any.whl

# --- 3. COPIA DE LOS ARCHIVOS DE EJECUCIÓN DE LA API ---
# Copiar el contenido de 'app/' (donde están main.py, api.py, config.py)
COPY zumba-churn-api/app/ ./app/

# Copiar el script de inicio
COPY zumba-churn-api/run.sh .

# --- 4. CONFIGURACIÓN DE SEGURIDAD Y EJECUCIÓN ---
RUN chmod +x ./run.sh
RUN chown -R api-user:api-user .
USER api-user

EXPOSE 8000

CMD ["bash", "./run.sh"]