FROM python:3.11-slim

# Se crea el directorio de trabajo.
WORKDIR /app

# Se copian e instlalan los requerimientos.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Se copi el código completo del Dashboard.
COPY . .

# Puerto del Dashboard.
EXPOSE 8050

# Se ejecuta el App con gunicorn.
CMD ["gunicorn", "app_dashboard:server", "--bind", "0.0.0.0:8050"]