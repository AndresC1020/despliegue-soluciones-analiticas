# Usa una imagen base oficial de Python
FROM python:3.12-slim

# --- 1. CONFIGURACIÓN DEL USUARIO NO-ROOT ---
RUN adduser --disabled-password --gecos '' api-user

# Definir directorio de trabajo dentro del contenedor
WORKDIR /opt/zumba-churn-api

# --- 2. INSTALACIÓN DE DEPENDENCIAS Y PAQUETE ---

# Copiar el archivo de dependencias primero para optimizar el cacheo.
COPY requirements/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install -r requirements.txt

# --- Instalación del paquete del modelo desde el archivo WHL ---
# 1. Copia el archivo .whl desde el directorio 'model-pkg' al WORKDIR.
COPY model-pkg/zumba_churn_pkg-0.0.1-py3-none-any.whl .

# 2. Instala el paquete usando pip.
RUN pip install zumba_churn_pkg-0.0.1-py3-none-any.whl

# --- Copiar los archivos de la API (FastAPI) ---
# Estos son los archivos de la aplicación que deben estar presentes para que Uvicorn se ejecute.
COPY config.yml .       # Archivo de configuración
COPY VERSION .          # Versión del modelo/API
COPY api.py .           # Lógica de los endpoints de la API
COPY config.py .        # Configuración de Pydantic/Logging
COPY main.py .          # Punto de entrada de FastAPI/Uvicorn

# --- 3. CONFIGURACIÓN DE SEGURIDAD Y EJECUCIÓN ---
# Copiar el script de inicio
COPY run.sh .

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