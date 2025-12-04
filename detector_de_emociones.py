import cv2
from keras.models import model_from_json
import numpy as np

# -------------------------------
# Cargar modelo
# -------------------------------
with open("emotiondetector.json", "r") as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

# -------------------------------
# Detector de caras
# -------------------------------
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# -------------------------------
# Función de extracción de features
# -------------------------------
def extract_features(image):
    feature = np.array(image, dtype=np.float32)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# -------------------------------
# Labels de emociones
# -------------------------------
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 
          4: 'neutral', 5: 'sad', 6: 'surprise'}

# -------------------------------
# Inicializar webcam
# -------------------------------
webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

frame_count = 0
last_faces = []
last_labels = []

try:
    while True:
        ret, frame = webcam.read()
        if not ret:
            break

        frame_count += 1
        display_frame = frame.copy()

        # -------------------------------
        # Procesar solo cada 3 frames
        # -------------------------------
        if frame_count % 3 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(30, 30)
            )
            last_faces = faces
            last_labels = []

            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                try:
                    face_img_resized = cv2.resize(face_img, (48,48))
                    img_features = extract_features(face_img_resized)
                    pred = model.predict(img_features, verbose=0)
                    prediction_label = labels[pred.argmax()]
                    last_labels.append(prediction_label)
                except:
                    last_labels.append("")

        # -------------------------------
        # Dibujar rectángulos y etiquetas
        # -------------------------------
        for i, (x, y, w, h) in enumerate(last_faces):
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            if i < len(last_labels):
                cv2.putText(display_frame, last_labels[i], (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # -------------------------------
        # Mostrar ventana
        # -------------------------------
        cv2.imshow("Emotion Detector", display_frame)

        # -------------------------------
        # Salir con 'q' o cerrando la ventana
        # -------------------------------
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if cv2.getWindowProperty("Emotion Detector", cv2.WND_PROP_VISIBLE) < 1:
            break

finally:
    webcam.release()
    cv2.destroyAllWindows()
