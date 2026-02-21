"""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║     API REST - DETECCIÓN DE NEUMONÍA CON IA                       ║
║     Sistema de predicción mediante Deep Learning                   ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
"""

# ════════════════════════════════════════════════════════════════════
# SECCIÓN 1: IMPORTACIONES
# ════════════════════════════════════════════════════════════════════

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import base64
from datetime import datetime
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════
# SECCIÓN 2: INICIALIZACIÓN DE FASTAPI
# ════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="API de Detección de Neumonía",
    description="API REST para clasificar radiografías de tórax como NORMAL o PNEUMONIA",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ════════════════════════════════════════════════════════════════════
# SECCIÓN 3: CARGAR MODELO
# ════════════════════════════════════════════════════════════════════

MODEL_DIR = Path(__file__).parent / ("models")
IMG_SIZE = (224, 224)

# Intentar cargar modelo en orden de preferencia
model = None
model_name = None

model_priority = [
    ('vgg16_finetuned.h5', 'VGG16 Fine-tuned'),
    ('vgg16_best.h5', 'VGG16 Best'),
    ('baseline_best.h5', 'CNN Baseline')
]

for filename, name in model_priority:
    model_path = MODEL_DIR / filename
    if model_path.exists():
        try:
            model = keras.models.load_model(model_path)
            model_name = name
            logger.info(f"✅ Modelo cargado: {name}")
            break
        except Exception as e:
            logger.error(f"❌ Error cargando {name}: {e}")
            continue

if model is None:
    logger.error("❌ No se pudo cargar ningún modelo")
else:
    logger.info(f"✅ API lista con modelo: {model_name}")

# ════════════════════════════════════════════════════════════════════
# SECCIÓN 4: FUNCIONES AUXILIARES
# ════════════════════════════════════════════════════════════════════

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocesa imagen para el modelo
    
    Args:
        image: Imagen PIL
        
    Returns:
        np.ndarray: Array normalizado (1, 224, 224, 3)
    """
    # Convertir a RGB si es necesario
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Redimensionar
    image = image.resize(IMG_SIZE)
    
    # Convertir a array y normalizar
    img_array = np.array(image)
    img_array = img_array.astype('float32') / 255.0
    
    # Agregar dimensión de batch
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def make_prediction(img_array: np.ndarray) -> dict:
    """
    Realiza predicción usando el modelo cargado
    
    Args:
        img_array: Imagen preprocesada
        
    Returns:
        dict: Resultado con predicción y probabilidades
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. Verifica carpeta 'models/'"
        )
    
    # Hacer predicción
    prediction = model.predict(img_array, verbose=0)[0][0]
    
    # Determinar clase
    predicted_class = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
    confidence = float(prediction if prediction > 0.5 else 1 - prediction)
    
    # Construir respuesta
    result = {
        "prediction": predicted_class,
        "confidence": confidence,
        "probabilities": {
            "NORMAL": float(1 - prediction),
            "PNEUMONIA": float(prediction)
        },
        "model_used": model_name,
        "timestamp": datetime.now().isoformat()
    }
    
    return result

# ════════════════════════════════════════════════════════════════════
# SECCIÓN 5: ENDPOINTS
# ════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    """Endpoint raíz - Información general"""
    return {
        "message": "API de Detección de Neumonía",
        "version": "1.0.0",
        "status": "online",
        "model": model_name if model else "No disponible",
        "endpoints": {
            "/": "Información general",
            "/health": "Estado del sistema",
            "/predict": "POST - Predicción con archivo",
            "/predict_base64": "POST - Predicción con base64",
            "/docs": "Documentación Swagger",
            "/redoc": "Documentación ReDoc"
        },
        "usage": {
            "example_curl": 'curl -X POST "http://api-url/predict" -F "file=@xray.jpg"',
            "example_python": 'requests.post("http://api-url/predict", files={"file": open("xray.jpg", "rb")})'
        }
    }

@app.get("/health")
async def health_check():
    """Endpoint de health check - Verifica estado"""
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_name": model_name if model else "None",
        "timestamp": datetime.now().isoformat(),
        "message": "API funcionando correctamente" if model else "API sin modelo cargado"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predicción con archivo de imagen
    
    Args:
        file: Archivo de imagen (JPG, PNG)
        
    Returns:
        JSON con predicción y probabilidades
    """
    try:
        # Validar tipo
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail=f"Archivo debe ser imagen. Recibido: {file.content_type}"
            )
        
        # Leer archivo
        contents = await file.read()
        
        # Validar tamaño (10 MB máx)
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Imagen muy grande. Máximo: 10 MB"
            )
        
        # Abrir imagen
        image = Image.open(io.BytesIO(contents))
        
        logger.info(f"📸 Imagen: {file.filename}, Tamaño: {image.size}")
        
        # Preprocesar
        img_array = preprocess_image(image)
        
        # Predecir
        result = make_prediction(img_array)
        
        logger.info(f"✅ Predicción: {result['prediction']} ({result['confidence']:.2%})")
        
        return JSONResponse(content=result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando imagen: {str(e)}"
        )

@app.post("/predict_base64")
async def predict_base64(data: dict):
    """
    Predicción con imagen en base64
    
    Args:
        data: {"image": "base64_string"}
        
    Returns:
        JSON con predicción
    """
    try:
        # Validar campo
        if "image" not in data:
            raise HTTPException(
                status_code=400,
                detail='Campo "image" requerido con string base64'
            )
        
        # Decodificar
        try:
            image_data = base64.b64decode(data["image"])
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error decodificando base64: {str(e)}"
            )
        
        # Abrir imagen
        image = Image.open(io.BytesIO(image_data))
        
        logger.info(f"📸 Imagen base64, Tamaño: {image.size}")
        
        # Preprocesar
        img_array = preprocess_image(image)
        
        # Predecir
        result = make_prediction(img_array)
        
        logger.info(f"✅ Predicción: {result['prediction']} ({result['confidence']:.2%})")
        
        return JSONResponse(content=result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando imagen: {str(e)}"
        )

# ════════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA (Solo para desarrollo local)
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║  API DE DETECCIÓN DE NEUMONÍA - SERVIDOR LOCAL                ║
    ╚════════════════════════════════════════════════════════════════╝
    
    Servidor: http://localhost:8000
    Documentación: http://localhost:8000/docs
    
    Presiona Ctrl+C para detener
    """)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)