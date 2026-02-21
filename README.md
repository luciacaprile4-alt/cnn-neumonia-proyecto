# 🫁 Detección de Neumonía con Deep Learning
### Proyecto 7 — Redes Neuronales Convolucionales

> Sistema de clasificación automática de radiografías de tórax para detectar neumonía, desarrollado con TensorFlow/Keras y desplegado como API REST.

---

## 📋 Descripción

Este proyecto desarrolla un modelo de Deep Learning capaz de clasificar radiografías de tórax como **NORMAL** o **PNEUMONIA**, utilizando técnicas de Transfer Learning con VGG16. El modelo fue entrenado con el dataset Chest X-Ray de Kaggle y expuesto mediante una API REST construida con FastAPI, desplegada en Render.

---

## 🗂️ Estructura del Proyecto

```
proyecto-7/
│
├── notebooks/
│   └── proyecto7_neumonia.ipynb     ← Notebook principal con todo el desarrollo
│
├── imagenes/
│   ├── eda/                         ← Gráficas del análisis exploratorio
│   ├── metricas/                    ← Curvas de entrenamiento, ROC, matrices
│   └── predicciones/                ← Ejemplos de predicciones
│
├── models/                          ← Modelos entrenados (ver enlaces abajo)
│
└── API/                             ← Código de la API REST
    ├── main.py
    ├── requirements.txt
    ├── Dockerfile
    └── .gitignore
```

---

## 🔗 Modelos Entrenados

Los modelos no están incluidos en el repositorio por su tamaño. Puedes descargarlos desde Google Drive:

| Modelo | Descripción | Enlace |
|--------|-------------|--------|
| `vgg16_finetuned.keras` | VGG16 con fine-tuning (modelo principal) | [Descargar](https://drive.google.com/file/d/1KfbTK9PHvh6xivp2ap2s-Yi2USmA656Z) |
| `baseline_best.h5` | CNN baseline sin transfer learning | *(agregar enlace)* |
| `mobilenet_pneumonia.h5` | MobileNetV2 fine-tuned | *(agregar enlace)* |

---

## 📊 Dataset

- **Fuente:** [Chest X-Ray Images (Pneumonia) — Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- **Clases:** NORMAL / PNEUMONIA
- **Distribución:** Train / Validation / Test
- **Preprocesamiento:** Normalización de píxeles (0-1), redimensionamiento a 224×224

---

## 🧠 Fases del Proyecto

### Fase 1 — Análisis Exploratorio (EDA)
- Exploración de la estructura del dataset y distribución de clases
- Análisis de dimensiones, calidad y variabilidad de imágenes
- Detección de imágenes corruptas o duplicadas
- Visualizaciones: histogramas de píxeles, distribución de clases, muestras por categoría

### Fase 2 — Entrenamiento de Modelos
- **Modelo baseline:** CNN simple como línea base de rendimiento
- **Data Augmentation:** rotaciones, zoom, flips, ajustes de brillo y contraste
- **Transfer Learning:** VGG16 y MobileNetV2 pre-entrenados con fine-tuning
- **Callbacks:** EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

### Fase 3 — Métricas y Tuning
- Curvas de entrenamiento (Loss y Accuracy — train vs validation)
- Matriz de confusión con colormap personalizado
- Curva ROC-AUC
- Métricas: Accuracy, Precision, Recall, F1-Score, AUC-ROC, Specificity, Sensitivity
- Tuning de hiperparámetros: learning rate, batch size, dropout, optimizadores
- Métodos ensemble

### Fase 4 — API REST
- Backend: FastAPI
- Endpoints: `/predict` (archivo), `/predict_base64` (base64), `/health`
- Deployment: Docker + Render
- Respuesta enriquecida con descripción, recomendación y nivel de confianza

### Fase 5 — Presentación
- Slides con descripción del problema, datos, solución, resultados e impacto médico

---

## 🚀 API en Producción

**URL Base:** `https://cnn-neumonia-api.onrender.com`

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/` | GET | Información general |
| `/health` | GET | Estado del sistema |
| `/predict` | POST | Predicción con archivo de imagen |
| `/predict_base64` | POST | Predicción con imagen en base64 |
| `/docs` | GET | Documentación Swagger interactiva |

### Ejemplo de respuesta

```json
{
  "titulo": "Resultado: Neumonía detectada",
  "descripcion": "La radiografía presenta patrones compatibles con neumonía.",
  "recomendacion": "Se recomienda consultar a un médico a la brevedad para confirmar el diagnóstico y recibir tratamiento.",
  "prediction": "PNEUMONIA",
  "confidence": "95.3%",
  "nivel_confianza": "Alta",
  "probabilities": {
    "NORMAL": "4.7%",
    "PNEUMONIA": "95.3%"
  }
}
```

### Ejemplo de uso en Python

```python
import requests

url = "https://cnn-neumonia-api.onrender.com/predict"
with open("radiografia.jpg", "rb") as f:
    response = requests.post(url, files={"file": f})

print(response.json())
```

---

## ⚙️ Instalación Local

```bash
# Clonar repositorio
git clone https://github.com/luciacaprile4-alt/cnn-neumonia-proyecto.git
cd cnn-neumonia-proyecto/API

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar API
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## 🛠️ Tecnologías Utilizadas

| Categoría                 | Tecnología                         |
|---------------------------|------------------------------------|
| Lenguaje                  | Python 3.9                         |
| Deep Learning             | TensorFlow 2.10 / Keras            |
| API                       | FastAPI + Uvicorn                  |
| Containerización          | Docker                             |
| Deployment                | Render                             |
| Almacenamiento de modelos | Google Drive + gdown               |
| Análisis de datos         | NumPy, Pandas, Matplotlib, Seaborn |
| Procesamiento de imágenes | Pillow, OpenCV                     |

---

## ⚠️ Aviso Médico

> Este sistema es una herramienta de apoyo al diagnóstico desarrollada con fines académicos. **No reemplaza el criterio de un médico especialista.** Ante cualquier resultado, se recomienda consultar con un profesional de la salud.

---

## 👩‍💻 Autora

**Lucía Caprile**
Proyecto desarrollado como parte del programa de formación en Data Science / Machine Learning, UNIVERSIDAD DEL DESARROLLO. COHORT 12.
