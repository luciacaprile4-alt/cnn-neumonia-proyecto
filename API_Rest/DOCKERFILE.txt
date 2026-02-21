# ════════════════════════════════════════════════════════════════════
# DOCKERFILE - API DE DETECCIÓN DE NEUMONÍA
# ════════════════════════════════════════════════════════════════════
#
# DESCRIPCIÓN:
# Dockerfile para containerizar la API y desplegarla en servicios como
# Render, Railway, Google Cloud Run, AWS ECS, etc.
#
# CARACTERÍSTICAS:
# - Imagen base: Python 3.11 slim (ligera)
# - Multi-stage build para optimizar tamaño
# - Dependencias del sistema para TensorFlow
# - Copia selectiva de archivos necesarios
#
# USO:
# docker build -t pneumonia-api .
# docker run -p 8000:8000 pneumonia-api
#
# ════════════════════════════════════════════════════════════════════

# ────────────────────────────────────────────────────────────────────
# ETAPA 1: Imagen base con Python
# ────────────────────────────────────────────────────────────────────

FROM python:3.11-slim as base

# Metadatos
LABEL maintainer="tu-email@example.com"
LABEL description="API REST para detección de neumonía con Deep Learning"
LABEL version="1.0.0"

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ────────────────────────────────────────────────────────────────────
# ETAPA 2: Instalar dependencias del sistema
# ────────────────────────────────────────────────────────────────────

# Instalar dependencias necesarias para TensorFlow y Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# ────────────────────────────────────────────────────────────────────
# ETAPA 3: Configurar directorio de trabajo
# ────────────────────────────────────────────────────────────────────

WORKDIR /app

# ────────────────────────────────────────────────────────────────────
# ETAPA 4: Instalar dependencias de Python
# ────────────────────────────────────────────────────────────────────

# Copiar solo requirements.txt primero (layer caching)
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# ────────────────────────────────────────────────────────────────────
# ETAPA 5: Copiar código de la aplicación
# ────────────────────────────────────────────────────────────────────

# Copiar archivos de la API
COPY main.py .

# Copiar modelos entrenados
COPY models/ models/

# Crear directorio para logs (opcional)
RUN mkdir -p logs

# ────────────────────────────────────────────────────────────────────
# ETAPA 6: Configurar usuario no-root (seguridad)
# ────────────────────────────────────────────────────────────────────

# Crear usuario no-root
RUN useradd -m -u 1000 apiuser && \
    chown -R apiuser:apiuser /app

# Cambiar a usuario no-root
USER apiuser

# ────────────────────────────────────────────────────────────────────
# ETAPA 7: Exponer puerto y comando de inicio
# ────────────────────────────────────────────────────────────────────

# Puerto que usará la aplicación
EXPOSE 8000

# Health check (Docker verificará que la API esté funcionando)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Comando para iniciar la aplicación
# Nota: En producción, ajustar workers según CPU disponibles
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# ════════════════════════════════════════════════════════════════════
# NOTAS DE USO:
# 
# BUILD:
#   docker build -t pneumonia-api .
# 
# RUN LOCAL:
#   docker run -p 8000:8000 pneumonia-api
# 
# RUN CON VOLUMEN (para desarrollo):
#   docker run -p 8000:8000 -v $(pwd)/models:/app/models pneumonia-api
#
# TAMAÑO FINAL ESPERADO: ~1.5-2 GB
# ════════════════════════════════════════════════════════════════════