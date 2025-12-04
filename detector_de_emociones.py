import cv2
from keras.models import model_from_json
import numpy as np


# Cargar modelo
with open("emotiondetector.json", "r") as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")


# Cargar detector de caras
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Función de extracción de features
def extract_features(image):
    feature = np.array(image, dtype=np.float32)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Labels de emociones
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 
          4: 'neutral', 5: 'sad', 6: 'surprise'}

# Inicializar camara
webcam = cv2.VideoCapture(0)
# Reducir resolución para mejorar velocidad
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

frame_count = 0

try:
    while True:
        ret, im = webcam.read()
        if not ret:
            print("Error: no se pudo capturar la imagen de la cámara")
            break

        frame_count += 1

        # Mostrar todos los frames sin procesar cada vez
        display_im = im.copy()

        # Procesar solo cada 2 frames
        if frame_count % 2 == 0:
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(30, 30)
            )

            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                cv2.rectangle(display_im, (x, y), (x+w, y+h), (255, 0, 0), 2)

                try:
                    face_img_resized = cv2.resize(face_img, (48,48))
                    img_features = extract_features(face_img_resized)
                    pred = model.predict(img_features)
                    prediction_label = labels[pred.argmax()]
                    cv2.putText(display_im, prediction_label, (x-10, y-10),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0,0,255), 2)
                except Exception as e:
                    print("Error procesando la cara:", e)
                    continue

        cv2.imshow("Emotion Detector", display_im)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Saliendo...")
            break

finally:
    webcam.release()
    cv2.destroyAllWindows()
