from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
import dlib
import joblib
import os

# Inicializar Flask
app = Flask(__name__)

# Cargar modelo SVM entrenado
modelo_svm = joblib.load("modelo_svm.pkl")

# Cargar herramientas de Dlib
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Ruta principal: Página para subir imágenes
@app.route('/')
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Reconocimiento Facial</title>
    </head>
    <body>
        <h1>Sube una imagen para reconocimiento facial</h1>
        <form action="/upload" method="post" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*" required>
            <button type="submit">Subir</button>
        </form>
    </body>
    </html>
    """

# Ruta para procesar la imagen
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No se encontró una imagen"}), 400

    file = request.files['image']
    file_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)

    # Procesar la imagen
    result = procesar_imagen(file_path)
    os.remove(file_path)  # Eliminar imagen después de procesarla

    return jsonify(result)

# Función para procesar la imagen
def procesar_imagen(image_path):
    img = cv2.imread(image_path)

    if img is None:
        return {"error": "No se pudo leer la imagen"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return {"message": "No se detectaron rostros en la imagen"}

    results = []

    for (x, y, w, h) in faces:
        # Recortar y convertir el rostro detectado
        face = img[y:y+h, x:x+w]
        rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        dlib_rect = dlib.rectangle(0, 0, face.shape[1], face.shape[0])

        # Generar embeddings
        shape = shape_predictor(rgb_face, dlib_rect)
        face_embedding = np.array(face_recognizer.compute_face_descriptor(rgb_face, shape)).reshape(1, -1)

        # Predecir con el modelo
        predicted_label = modelo_svm.predict(face_embedding)[0]
        predicted_prob = modelo_svm.predict_proba(face_embedding).max()

        results.append({
            "nombre": predicted_label,
            "probabilidad": f"{predicted_prob * 100:.2f}%",
            "ubicacion": {"x": int(x), "y": int(y), "ancho": int(w), "alto": int(h)}
        })

    return {"message": "Rostros detectados", "resultados": results}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
