# Usa una imagen base oficial de Python
FROM python:3.12-slim

# --- 1. CONFIGURACIÓN DEL USUARIO NO-ROOT ---
# Crear usuario que ejecuta la app
RUN adduser --disabled-password --gecos '' api-user

# Definir directorio de trabajo dentro del contenedor
WORKDIR /opt/zumba-churn-api

# --- 2. INSTALACIÓN DE DEPENDENCIAS Y PAQUETE ---

# Copiar el archivo de dependencias (Ruta desde el Contexto Raíz)
COPY requirements/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install -r requirements.txt

# --- Instalación del paquete del modelo desde el archivo WHL ---
# 1. Copia el archivo .whl (Ruta desde el Contexto Raíz)
COPY model-pkg/zumba_churn_pkg-0.0.1-py3-none-any.whl .

# 2. Instala el paquete usando pip.
RUN pip install zumba_churn_pkg-0.0.1-py3-none-any.whl

# --- Copiar los archivos de la API (FastAPI) ---
# ¡ATENCIÓN! Ahora se usa la ruta anidada: zumba_churn_api-docker/zumba-churn-api/
COPY zumba_churn_api-docker/zumba-churn-api/config.yml .
COPY zumba_churn_api-docker/zumba-churn-api/VERSION .
COPY zumba_churn_api-docker/zumba-churn-api/api.py .
COPY zumba_churn_api-docker/zumba-churn-api/config.py .
COPY zumba_churn_api-docker/zumba-churn-api/main.py .

# --- 3. CONFIGURACIÓN DE SEGURIDAD Y EJECUCIÓN ---
# Copiar el script de inicio (Ruta anidada)
COPY zumba_churn_api-docker/zumba-churn-api/run.sh .

# Hacer el script de inicio ejecutable
RUN chmod +x ./run.sh
# Cambiar propiedad de la carpeta al usuario no-root
RUN chown -R api-user:api-user .

# Cambiar al usuario no-root para ejecutar la aplicación
USER api-user

# Puerto a exponer para la api
EXPOSE 8000

# Comandos a ejecutar al correr el contenedor
CMD ["bash", "./run.sh"]