# Imagen base ligera de Python
FROM python:3.12-slim

# Crear usuario no root para ejecutar la app
RUN adduser --disabled-password --gecos '' api-user

# Definir directorio de trabajo
WORKDIR /opt/zumba-churn-api

# Copiar el código de la aplicación dentro del contenedor
COPY ./zumba-churn-api /opt/zumba-churn-api/

# Actualizar pip e instalar dependencias
RUN pip install --upgrade pip
RUN pip install -r /opt/zumba-churn-api/requirements.txt

# Dar permisos de ejecución al script de arranque
RUN chmod +x /opt/zumba-churn-api/run.sh

# Cambiar propiedad de la carpeta al usuario api-user
RUN chown -R api-user:api-user /opt/zumba-churn-api

# Cambiar al usuario no root
USER api-user

ENV PORT=8000

# Exponer el puerto de la API
EXPOSE 8000

# Comando por defecto al iniciar el contenedor
CMD ["bash", "./run.sh"]
