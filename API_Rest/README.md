# 🏥 API de Detección de Neumonía con Deep Learning

API REST para clasificar radiografías de tórax como **NORMAL** o **PNEUMONIA** usando modelos de Deep Learning.

---

## 📋 Tabla de Contenidos

- [Características](#-características)
- [Instalación](#-instalación)
- [Uso Local](#-uso-local)
- [Endpoints](#-endpoints)
- [Deployment](#-deployment)
- [Testing](#-testing)
- [Troubleshooting](#-troubleshooting)

---

## ✨ Características

- ✅ **Predicción en tiempo real** (2-3 segundos)
- ✅ **Múltiples formatos de entrada** (archivo o base64)
- ✅ **Documentación interactiva** (Swagger UI)
- ✅ **CORS habilitado** para aplicaciones web
- ✅ **Manejo robusto de errores**
- ✅ **Logging detallado**

---

## 🚀 Instalación

### Prerrequisitos

- Python 3.11+
- Modelo entrenado (`.keras`) en carpeta `models/`

### Pasos
```bash
# 1. Clonar repositorio (o descargar archivos)
git clone https://github.com/tu-usuario/pneumonia-api.git
cd pneumonia-api

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Verificar que el modelo esté en models/
ls models/  # Debe contener vgg16_finetuned.keras (o similar)
```

---

## 💻 Uso Local

### Iniciar servidor
```bash
uvicorn main:app --reload
```

La API estará disponible en: **http://localhost:8000**

### Acceder a documentación

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 📡 Endpoints

### 1. **GET /** - Información General
```bash
curl http://localhost:8000/
```

**Respuesta:**
```json
{
  "message": "API de Detección de Neumonía",
  "version": "1.0.0",
  "status": "online",
  "model": "VGG16 Fine-tuned"
}
```

---

### 2. **GET /health** - Estado del Sistema
```bash
curl http://localhost:8000/health
```

**Respuesta:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "VGG16 Fine-tuned",
  "timestamp": "2026-02-17T10:30:00"
}
```

---

### 3. **POST /predict** - Predicción con Archivo

**cURL:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/xray.jpg"
```

**Python:**
```python
import requests

with open('xray.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict',
        files={'file': f}
    )

print(response.json())
```

**JavaScript:**
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/predict', {
  method: 'POST',
  body: formData
})
.then(res => res.json())
.then(data => console.log(data));
```

**Respuesta:**
```json
{
  "prediction": "PNEUMONIA",
  "confidence": 0.94,
  "probabilities": {
    "NORMAL": 0.06,
    "PNEUMONIA": 0.94
  },
  "model_used": "VGG16 Fine-tuned",
  "timestamp": "2026-02-17T10:30:00"
}
```

---

### 4. **POST /predict_base64** - Predicción con Base64

**Python:**
```python
import requests
import base64

with open('xray.jpg', 'rb') as f:
    image_b64 = base64.b64encode(f.read()).decode('utf-8')

response = requests.post(
    'http://localhost:8000/predict_base64',
    json={'image': image_b64}
)

print(response.json())
```

**Respuesta:** Igual que `/predict`

---

## 🌐 Deployment

### Opción 1: Render (Recomendado)

1. **Crear cuenta**: https://render.com
2. **New Web Service** → Connect GitHub
3. **Configuración:**
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. **Subir modelos** a carpeta `models/`
5. **Deploy**

URL resultante: `https://tu-app.onrender.com`

---

### Opción 2: Hugging Face Spaces

1. **Crear Space**: https://huggingface.co/spaces
2. **Configuración:**
   - SDK: Gradio o Docker
   - Subir `main.py`, `requirements.txt`, `models/`
3. **Deploy automático**

URL resultante: `https://tu-usuario-pneumonia-api.hf.space`

---

## 🧪 Testing

### Ejecutar tests
```bash
# 1. Crear carpeta de imágenes de prueba
mkdir test_images

# 2. Colocar imágenes .jpg/.png en test_images/

# 3. Ejecutar tests
python test_api.py
```

### Salida esperada
```
╔══════════════════════════════════════════════════════════════════╗
║                    TEST SUITE - API NEUMONÍA                     ║
╚══════════════════════════════════════════════════════════════════╝

TEST 1: HEALTH CHECK
✅ Status: 200
✅ API Status: healthy
✅ Modelo cargado: True

TEST 2: PREDICCIÓN CON ARCHIVO
✅ Predicción: PNEUMONIA
✅ Confianza: 94%

RESUMEN DE TESTS
✅ Health Check: PASSED
✅ Predict File: PASSED
✅ Predict Base64: PASSED
✅ Error Handling: PASSED

🎉 TODOS LOS TESTS PASARON
```

---

## 🔧 Troubleshooting

### Problema: "Modelo no disponible"

**Causa:** Archivos `.keras` no están en `models/`

**Solución:**
```bash
# Verificar que existan los modelos
ls models/

# Debe contener al menos uno de:
# - vgg16_finetuned.keras
# - vgg16_best.keras
# - baseline_best.keras
```

---

### Problema: "ModuleNotFoundError: No module named 'tensorflow'"

**Causa:** TensorFlow no instalado

**Solución:**
```bash
pip install tensorflow==2.16.1
```

---

### Problema: API muy lenta

**Causa:** Ejecutando en CPU

**Soluciones:**
- **Opción A:** Usar servicio con GPU (Hugging Face Spaces con GPU)
- **Opción B:** Optimizar modelo (quantización)
- **Opción C:** Aceptar 2-3 segundos de latencia

---

### Problema: "Connection refused"

**Causa:** Servidor no está corriendo

**Solución:**
```bash
# Verificar que el servidor esté activo
uvicorn main:app --reload

# Verificar puerto correcto
curl http://localhost:8000/health
```

---

## 📞 Soporte

- **Documentación interactiva**: http://localhost:8000/docs
- **Issues**: [GitHub Issues]
- **Email**: tu-email@example.com

---

## 📄 Licencia

MIT License - Ver `LICENSE` para más detalles

---

## 🙏 Agradecimientos

- Dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Framework: [FastAPI](https://fastapi.tiangolo.com/)
- ML Framework: [TensorFlow](https://www.tensorflow.org/)

---

**Desarrollado con ❤️ para ayudar en el diagnóstico médico**