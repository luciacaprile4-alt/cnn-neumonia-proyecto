"""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║     CLIENTE DE PRUEBA - API DE DETECCIÓN DE NEUMONÍA             ║
║     Script para testear todos los endpoints de la API             ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝

DESCRIPCIÓN:
Este script prueba todos los endpoints de la API para verificar
que funcionen correctamente antes del deployment.

FUNCIONES:
1. test_health() - Verifica estado de la API
2. test_predict_file() - Prueba predicción con archivo
3. test_predict_base64() - Prueba predicción con base64
4. test_invalid_inputs() - Prueba manejo de errores

USO:
1. Asegúrate de que la API esté corriendo (uvicorn main:app)
2. Coloca imágenes de prueba en la carpeta 'test_images/'
3. Ejecuta: python test_api.py

REQUISITOS:
- pip install requests
- API corriendo en http://localhost:8000
- Imágenes de prueba disponibles
"""

# ════════════════════════════════════════════════════════════════════
# IMPORTACIONES
# ════════════════════════════════════════════════════════════════════

import requests
import base64
from pathlib import Path
import json
import sys

# ════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════════

# URL de la API (cambiar según deployment)
API_URL = "http://localhost:8000"  # Local
# API_URL = "https://tu-app.onrender.com"  # Render
# API_URL = "https://tu-usuario-pneumonia-api.hf.space"  # Hugging Face

# Colores para terminal
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_test(message, status="info"):
    """Imprime mensajes de test con colores"""
    colors = {
        "info": Colors.OKBLUE,
        "success": Colors.OKGREEN,
        "warning": Colors.WARNING,
        "error": Colors.FAIL,
        "header": Colors.HEADER
    }
    color = colors.get(status, Colors.ENDC)
    print(f"{color}{message}{Colors.ENDC}")

# ════════════════════════════════════════════════════════════════════
# TEST 1: HEALTH CHECK
# ════════════════════════════════════════════════════════════════════

def test_health():
    """
    Prueba el endpoint /health
    
    VERIFICA:
    - API está online
    - Modelo está cargado
    - Respuesta tiene formato correcto
    """
    print_test("\n" + "="*70, "header")
    print_test("TEST 1: HEALTH CHECK", "header")
    print_test("="*70, "header")
    
    try:
        response = requests.get(f"{API_URL}/health")
        
        if response.status_code == 200:
            data = response.json()
            print_test(f"✅ Status: {response.status_code}", "success")
            print_test(f"✅ API Status: {data.get('status')}", "success")
            print_test(f"✅ Modelo cargado: {data.get('model_loaded')}", "success")
            print_test(f"✅ Modelo: {data.get('model_name')}", "success")
            
            if not data.get('model_loaded'):
                print_test("⚠️ ADVERTENCIA: Modelo no está cargado", "warning")
                return False
            
            return True
        else:
            print_test(f"❌ Error: Status {response.status_code}", "error")
            return False
            
    except requests.exceptions.ConnectionError:
        print_test("❌ Error: No se pudo conectar a la API", "error")
        print_test(f"   Verifica que esté corriendo en {API_URL}", "error")
        return False
    except Exception as e:
        print_test(f"❌ Error inesperado: {e}", "error")
        return False

# ════════════════════════════════════════════════════════════════════
# TEST 2: PREDICCIÓN CON ARCHIVO
# ════════════════════════════════════════════════════════════════════

def test_predict_file(image_path):
    """
    Prueba el endpoint /predict con un archivo de imagen
    
    Args:
        image_path: Ruta a la imagen de prueba
        
    VERIFICA:
    - Archivo se envía correctamente
    - Predicción se realiza
    - Respuesta tiene formato esperado
    - Confianza está en rango [0, 1]
    """
    print_test("\n" + "="*70, "header")
    print_test(f"TEST 2: PREDICCIÓN CON ARCHIVO - {image_path}", "header")
    print_test("="*70, "header")
    
    if not Path(image_path).exists():
        print_test(f"❌ Archivo no encontrado: {image_path}", "error")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (Path(image_path).name, f, 'image/jpeg')}
            response = requests.post(f"{API_URL}/predict", files=files)
        
        if response.status_code == 200:
            data = response.json()
            
            print_test(f"✅ Status: {response.status_code}", "success")
            print_test(f"✅ Predicción: {data.get('prediction')}", "success")
            print_test(f"✅ Confianza: {data.get('confidence'):.2%}", "success")
            print_test(f"✅ Prob NORMAL: {data.get('probabilities', {}).get('NORMAL'):.2%}", "success")
            print_test(f"✅ Prob PNEUMONIA: {data.get('probabilities', {}).get('PNEUMONIA'):.2%}", "success")
            print_test(f"✅ Modelo usado: {data.get('model_used')}", "success")
            print_test(f"✅ Timestamp: {data.get('timestamp')}", "success")
            
            # Validaciones
            confidence = data.get('confidence', 0)
            if not (0 <= confidence <= 1):
                print_test("⚠️ Confianza fuera de rango [0, 1]", "warning")
            
            return True
        else:
            print_test(f"❌ Error: Status {response.status_code}", "error")
            print_test(f"   Detalle: {response.text}", "error")
            return False
            
    except Exception as e:
        print_test(f"❌ Error: {e}", "error")
        return False

# ════════════════════════════════════════════════════════════════════
# TEST 3: PREDICCIÓN CON BASE64
# ════════════════════════════════════════════════════════════════════

def test_predict_base64(image_path):
    """
    Prueba el endpoint /predict_base64 con imagen en base64
    
    Args:
        image_path: Ruta a la imagen de prueba
        
    VERIFICA:
    - Codificación base64 funciona
    - Predicción se realiza correctamente
    - Respuesta es consistente con /predict
    """
    print_test("\n" + "="*70, "header")
    print_test(f"TEST 3: PREDICCIÓN CON BASE64 - {image_path}", "header")
    print_test("="*70, "header")
    
    if not Path(image_path).exists():
        print_test(f"❌ Archivo no encontrado: {image_path}", "error")
        return False
    
    try:
        # Leer y codificar imagen
        with open(image_path, 'rb') as f:
            image_b64 = base64.b64encode(f.read()).decode('utf-8')
        
        # Enviar petición
        response = requests.post(
            f"{API_URL}/predict_base64",
            json={"image": image_b64}
        )
        
        if response.status_code == 200:
            data = response.json()
            
            print_test(f"✅ Status: {response.status_code}", "success")
            print_test(f"✅ Predicción: {data.get('prediction')}", "success")
            print_test(f"✅ Confianza: {data.get('confidence'):.2%}", "success")
            
            return True
        else:
            print_test(f"❌ Error: Status {response.status_code}", "error")
            print_test(f"   Detalle: {response.text}", "error")
            return False
            
    except Exception as e:
        print_test(f"❌ Error: {e}", "error")
        return False

# ════════════════════════════════════════════════════════════════════
# TEST 4: MANEJO DE ERRORES
# ════════════════════════════════════════════════════════════════════

def test_invalid_inputs():
    """
    Prueba el manejo de entradas inválidas
    
    VERIFICA:
    - Archivo no-imagen es rechazado
    - Base64 inválido es manejado
    - Errores retornan códigos apropiados
    """
    print_test("\n" + "="*70, "header")
    print_test("TEST 4: MANEJO DE ERRORES", "header")
    print_test("="*70, "header")
    
    tests_passed = 0
    total_tests = 3
    
    # Test 4.1: Enviar archivo no-imagen
    print_test("\n4.1 - Enviar archivo de texto (debe fallar):", "info")
    try:
        files = {'file': ('test.txt', b'not an image', 'text/plain')}
        response = requests.post(f"{API_URL}/predict", files=files)
        
        if response.status_code == 400:
            print_test("✅ Error 400 retornado correctamente", "success")
            tests_passed += 1
        else:
            print_test(f"❌ Se esperaba 400, recibido {response.status_code}", "error")
    except Exception as e:
        print_test(f"❌ Error: {e}", "error")
    
    # Test 4.2: Base64 inválido
    print_test("\n4.2 - Enviar base64 inválido (debe fallar):", "info")
    try:
        response = requests.post(
            f"{API_URL}/predict_base64",
            json={"image": "invalid_base64!!!"}
        )
        
        if response.status_code == 400:
            print_test("✅ Error 400 retornado correctamente", "success")
            tests_passed += 1
        else:
            print_test(f"❌ Se esperaba 400, recibido {response.status_code}", "error")
    except Exception as e:
        print_test(f"❌ Error: {e}", "error")
    
    # Test 4.3: Campo faltante en base64
    print_test("\n4.3 - JSON sin campo 'image' (debe fallar):", "info")
    try:
        response = requests.post(
            f"{API_URL}/predict_base64",
            json={"wrong_field": "data"}
        )
        
        if response.status_code == 400:
            print_test("✅ Error 400 retornado correctamente", "success")
            tests_passed += 1
        else:
            print_test(f"❌ Se esperaba 400, recibido {response.status_code}", "error")
    except Exception as e:
        print_test(f"❌ Error: {e}", "error")
    
    print_test(f"\n✅ Tests de error pasados: {tests_passed}/{total_tests}", "success")
    return tests_passed == total_tests

# ════════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL
# ════════════════════════════════════════════════════════════════════

def main():
    """
    Ejecuta todos los tests de la API
    
    ORDEN:
    1. Health check
    2. Predicción con archivos
    3. Predicción con base64
    4. Manejo de errores
    
    RESULTADO:
    Muestra resumen de tests pasados/fallados
    """
    print_test("\n" + "╔" + "="*68 + "╗", "header")
    print_test("║" + " "*20 + "TEST SUITE - API NEUMONÍA" + " "*23 + "║", "header")
    print_test("╚" + "="*68 + "╝", "header")
    
    results = []
    
    # Test 1: Health Check
    results.append(("Health Check", test_health()))
    
    if not results[0][1]:
        print_test("\n❌ API no está disponible. Abortando tests.", "error")
        sys.exit(1)
    
    # Test 2 y 3: Predicciones
    # Buscar imágenes de prueba
    test_images_dir = Path("test_images")
    
    if test_images_dir.exists():
        test_images = list(test_images_dir.glob("*.jpeg")) + \
                     list(test_images_dir.glob("*.jpg")) + \
                     list(test_images_dir.glob("*.png"))
        
        if test_images:
            # Probar con primera imagen
            test_image = test_images[0]
            results.append(("Predict File", test_predict_file(str(test_image))))
            results.append(("Predict Base64", test_predict_base64(str(test_image))))
        else:
            print_test("\n⚠️ No se encontraron imágenes de prueba", "warning")
            results.append(("Predict File", None))
            results.append(("Predict Base64", None))
    else:
        print_test("\n⚠️ Carpeta 'test_images/' no existe", "warning")
        print_test("   Crea la carpeta y coloca imágenes de prueba", "warning")
        results.append(("Predict File", None))
        results.append(("Predict Base64", None))
    
    # Test 4: Manejo de errores
    results.append(("Error Handling", test_invalid_inputs()))
    
    # Resumen
    print_test("\n" + "="*70, "header")
    print_test("RESUMEN DE TESTS", "header")
    print_test("="*70, "header")
    
    passed = sum(1 for _, result in results if result is True)
    failed = sum(1 for _, result in results if result is False)
    skipped = sum(1 for _, result in results if result is None)
    total = len(results)
    
    for test_name, result in results:
        if result is True:
            print_test(f"✅ {test_name}: PASSED", "success")
        elif result is False:
            print_test(f"❌ {test_name}: FAILED", "error")
        else:
            print_test(f"⊘  {test_name}: SKIPPED", "warning")
    
    print_test("\n" + "-"*70, "info")
    print_test(f"Total: {total} | Passed: {passed} | Failed: {failed} | Skipped: {skipped}", "info")
    
    if failed == 0 and passed > 0:
        print_test("\n🎉 TODOS LOS TESTS PASARON", "success")
        return 0
    else:
        print_test("\n⚠️ ALGUNOS TESTS FALLARON", "warning")
        return 1

# ════════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    sys.exit(main())