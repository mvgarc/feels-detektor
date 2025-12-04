# Feels‑Detektor  

Detector de emociones faciales en tiempo real usando Python, OpenCV y una red neuronal.  

##  Qué hace ?

- Usa la cámara web para capturar video.  
- Detecta rostros con un clasificador Haar cascade.  
- Preprocesa el rostro a escala de grises y 48×48 px.  
- Utiliza un modelo entrenado (formato `.json` + `.h5`) para predecir la emoción:  
  - angry, disgust, fear, happy, neutral, sad, surprise  
- Muestra, en una ventana redimensionable, el video con rectángulos en los rostros y la emoción reconocida encima.  
- Permite cerrar la ventana presionando `q` o con la X correctamente.  

## 🚀 Características principales  

- Detector en **tiempo real** usando la webcam  
- Compatible con CPU (no requiere GPU)  
- Ventana redimensionable (`WINDOW_NORMAL`) y configurable en tamaño  
- Procesamiento de video optimizado: analiza solo cada N frames para reducir carga  
- Normalización de imágenes, soporte para múltiples rostros  

## 📁 Estructura del repositorio  
```bash
feels-detektor/
│
├── emotiondetector.json # Arquitectura del modelo
├── emotiondetector.h5 # Pesos del modelo entrenado
├── detector_de_emociones.py # Script principal para detección en tiempo real
├── README.md # Este archivo
└── (otros archivos de proyecto…)

```


## 🧰 Requisitos / Dependencias  

- Python 3.10 (recomendado)  
- OpenCV (`opencv-python`)  
- NumPy  
- TensorFlow / Keras  

Para instalarlas fácilmente, activa tu entorno virtual y luego:

```bash
pip install opencv-python numpy tensorflow keras
```

📥 Cómo ejecutar

Clona este repositorio:
```bash
git clone https://github.com/mvgarc/feels-detektor.git
cd feels-detektor
```
Asegúrate de usar Python 3.10 y tener un entorno virtual activado.

Instala las dependencias (ver sección anterior).

Ejecuta el script principal:

```bash
python detector_de_emociones.py
```

Se abrirá una ventana con la cámara. Presiona q o cierra la ventana para salir.

🔧 Cómo entrenar / reentrenar el modelo

Si quieres mejorar la precisión — especialmente para clases como “sad” o “angry” — te recomiendo:

Preparar un dataset balanceado con suficientes imágenes por emoción; todas en 48×48 px en escala de grises.

Aplicar Data Augmentation para ampliar el dataset.

Usar tu propio script de entrenamiento (por ejemplo con Keras), luego generar nuevos archivos emotiondetector.json + emotiondetector.h5.

Reemplazar los archivos en este repositorio por los nuevos.

📝 Licencia

Este proyecto está bajo licencia MIT — eres libre de usar, modificar y distribuir el código como desees.

✨ Agradecimientos / Referencias

Basado en técnicas comunes de detección facial con Haar cascade + redes neuronales para reconocimiento de emociones.

Inspirado en múltiples proyectos de visión por computadora usando Python + OpenCV.