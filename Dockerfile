# Imagen base ligera de Python
FROM python:3.12-slim

# Establecer directorio de trabajo
WORKDIR /app

# Copiar solo los archivos necesarios
COPY . /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    libpq-dev \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalar las dependencias de Python
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto de la app
EXPOSE 8000

# Comando para ejecutar la aplicación con Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
