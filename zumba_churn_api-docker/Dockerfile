FROM python:3.12-slim

# Crear usuario no-root por seguridad
RUN adduser --disabled-password --gecos '' api-user

# Definir el directorio de trabajo dentro del contenedor
WORKDIR /opt/zumba-churn-api

# --- 1. COPIAR TODOS LOS ARCHIVOS NECESARIOS ---
COPY zumba-churn-api/ .

# --- 2. INSTALACIÓN DE DEPENDENCIAS Y PAQUETE (.WHL) ---
RUN pip install --no-cache-dir --upgrade pip
RUN pip install -r requirements.txt

# --- 3. CONFIGURACIÓN DE SEGURIDAD Y EJECUCIÓN ---
RUN chmod +x ./run.sh
RUN chown -R api-user:api-user .
USER api-user

EXPOSE 8000

CMD ["bash", "./run.sh"]