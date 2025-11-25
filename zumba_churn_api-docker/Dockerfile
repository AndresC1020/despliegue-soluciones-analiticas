# Usa una imagen base oficial de Python
FROM python:3.12-slim

# --- 1. CONFIGURACIÓN DEL USUARIO NO-ROOT ---
# Crear usuario que ejecuta la app
RUN adduser --disabled-password --gecos '' api-user

# Definir directorio de trabajo
WORKDIR /opt/zumba-churn-api

# --- 2. INSTALACIÓN DE DEPENDENCIAS Y PAQUETE ---
# Copiar el archivo de dependencias y la carpeta del proyecto
COPY requirements/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install -r requirements.txt

# Copiar el paquete completo y archivos de la API
# Se copia todo lo necesario para la instalación y el runtime.
COPY setup.py .
COPY MANIFEST.in .
COPY VERSION .
COPY api.py .
COPY requirements/ ./requirements/
COPY zumba_churn_pkg ./zumba_churn_pkg

# Instalar su paquete localmente (necesario para que 'from zumba_churn_pkg...' funcione)
RUN pip install .

# --- 3. CONFIGURACIÓN DE SEGURIDAD Y EJECUCIÓN ---
# Copiar el script de inicio
COPY run.sh .

# Hacer el script de inicio ejecutable
RUN chmod +x ./run.sh
# Cambiar propiedad de la carpeta al usuario no-root
RUN chown -R api-user:api-user .

# Cambiar al usuario no-root para ejecutar la aplicación
USER api-user

# Puerto a exponer para la api (Usaremos el 8000 por defecto de Uvicorn/FastAPI)
EXPOSE 8000

# Comandos a ejecutar al correr el contenedor
CMD ["bash", "run.sh"]