import os
import cv2
import numpy as np
import glob
import dlib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import csv
from datetime import datetime

# Funciones del menú
def capturar_imagenes():
    """Captura imágenes para un nuevo usuario."""
    user_name = input("Introduce el nombre del usuario: ")
    output_dir = f'dataset/{user_name}'

    if os.path.exists(output_dir):
        print(f"Ya existe un usuario con el nombre '{user_name}'. Usa otro nombre o elimina al usuario existente.")
        return

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    image_count = 0
    print("Presiona 's' para capturar una imagen y 'q' para salir.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo acceder a la cámara.")
            break

        cv2.imshow("Captura de Rostro", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('s'):
            image_path = os.path.join(output_dir, f"{user_name}_{image_count}.jpg")
            cv2.imwrite(image_path, frame)
            image_count += 1
            print(f"Imagen {image_count} guardada en {image_path}")
        elif key & 0xFF == ord('q'):
            print("Saliendo de la captura.")
            break

    cap.release()
    cv2.destroyAllWindows()

def procesar_rostros():
    """Detecta y recorta rostros en las imágenes de todos los usuarios registrados."""
    print("Buscando usuarios registrados...")
    user_dirs = [dir for dir in os.listdir('dataset') if os.path.isdir(f'dataset/{dir}')]

    if not user_dirs:
        print("No se encontraron usuarios registrados. Asegúrate de capturar imágenes primero.")
        return

    print(f"Usuarios encontrados: {', '.join(user_dirs)}")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for user_name in user_dirs:
        input_dir = f'dataset/{user_name}'
        faces_dir = f'{input_dir}_faces'

        os.makedirs(faces_dir, exist_ok=True)

        print(f"Procesando imágenes para el usuario '{user_name}'...")
        for img_name in os.listdir(input_dir):
            img_path = os.path.join(input_dir, img_name)
            img = cv2.imread(img_path)

            if img is None:
                print(f"No se pudo leer la imagen {img_name}.")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for i, (x, y, w, h) in enumerate(faces):
                face = img[y:y+h, x:x+w]
                face_path = os.path.join(faces_dir, f"{img_name.split('.')[0]}_face_{i}.jpg")
                cv2.imwrite(face_path, face)
                print(f"Rostro detectado y guardado en {face_path}")

        print(f"Procesamiento para el usuario '{user_name}' completado.")

    print("Procesamiento de rostros finalizado para todos los usuarios registrados.")

def generar_embeddings_usuario():
    """Genera embeddings faciales para un usuario específico y los guarda por separado."""
    user_name = input("Introduce el nombre del usuario: ")
    faces_dir = f'dataset/{user_name}_faces'

    if not os.path.exists(faces_dir):
        print(f"No se encontraron rostros procesados para el usuario '{user_name}'. Procesa los rostros primero.")
        return

    # Inicializar herramientas de dlib
    shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

    embeddings = []
    labels = []

    for img_name in os.listdir(faces_dir):
        img_path = os.path.join(faces_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"No se pudo leer la imagen {img_name}.")
            continue

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dlib_rect = dlib.rectangle(0, 0, img.shape[1], img.shape[0])
        shape = shape_predictor(rgb_img, dlib_rect)
        face_embedding = np.array(face_recognizer.compute_face_descriptor(rgb_img, shape))

        embeddings.append(face_embedding)
        labels.append(user_name)

    if embeddings:
        embeddings = np.array(embeddings)
        labels = np.array(labels)
        np.save(f"embeddings_{user_name}.npy", embeddings)
        np.save(f"labels_{user_name}.npy", labels)
        print(f"Embeddings y etiquetas para '{user_name}' guardados en archivos separados.")
    else:
        print(f"No se generaron embeddings para el usuario '{user_name}'. Verifica los datos.")

def entrenar_modelo():
    """Entrena un modelo SVM usando todos los embeddings generados."""
    embeddings = []
    labels = []

    for emb_file in glob.glob("embeddings_*.npy"):
        user_embeddings = np.load(emb_file)
        user_labels = np.load(emb_file.replace("embeddings_", "labels_"))
        embeddings.append(user_embeddings)
        labels.append(user_labels)

    if not embeddings or not labels:
        print("Error: No se encontraron datos para entrenar el modelo. Asegúrate de generar embeddings primero.")
        return

    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)

    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

    clf = SVC(kernel='rbf', probability=True, class_weight='balanced')
    clf.fit(X_train, y_train)
    print("Modelo entrenado con éxito.")

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisión del modelo: {accuracy * 100:.2f}%")
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred, zero_division=0))

    joblib.dump(clf, 'modelo_svm.pkl')
    print("Modelo guardado como 'modelo_svm.pkl'.")

def registrar_acceso(evento):
    """Registra entrada o salida de usuarios."""
    try:
        clf = joblib.load('modelo_svm.pkl')
    except FileNotFoundError:
        print("No se encontró el modelo entrenado. Entrena el modelo primero.")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

    log_file = "accesos.csv"
    if not os.path.exists(log_file):
        with open(log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Nombre", "Fecha", "Hora", "Evento"])

    cap = cv2.VideoCapture(0)
    print("Presiona 'q' para salir.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo acceder a la cámara.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            roi = frame[y:y+h, x:x+w]
            rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            dlib_rect = dlib.rectangle(left=x, top=y, right=x+w, bottom=y+h)

            shape = shape_predictor(rgb_roi, dlib_rect)
            face_embedding = np.array(face_recognizer.compute_face_descriptor(rgb_roi, shape)).reshape(1, -1)

            predicted_label = clf.predict(face_embedding)[0]
            now = datetime.now()
            date = now.strftime("%Y-%m-%d")
            time = now.strftime("%H:%M:%S")

            with open(log_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([predicted_label, date, time, evento])

            print(f"{evento} registrada para {predicted_label} a las {time}.")

        cv2.imshow(f"Registro de {evento}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def eliminar_usuario():
    """Elimina los datos de un usuario, incluyendo imágenes y directorios procesados."""
    user_name = input("Introduce el nombre del usuario a eliminar: ")
    input_dir = f'dataset/{user_name}'
    faces_dir = f'{input_dir}_faces'

    # Eliminar imágenes originales
    if os.path.exists(input_dir):
        for file in os.listdir(input_dir):
            os.remove(os.path.join(input_dir, file))
        os.rmdir(input_dir)  # Eliminar el directorio vacío
        print(f"Se han eliminado las imágenes originales del usuario '{user_name}'.")
    else:
        print(f"No se encontraron imágenes originales para el usuario '{user_name}'.")

    # Eliminar imágenes procesadas (_faces)
    if os.path.exists(faces_dir):
        for file in os.listdir(faces_dir):
            os.remove(os.path.join(faces_dir, file))
        os.rmdir(faces_dir)  # Eliminar el directorio vacío
        print(f"Se han eliminado las imágenes procesadas del usuario '{user_name}' (_faces).")
    else:
        print(f"No se encontraron imágenes procesadas para el usuario '{user_name}'.")

    # Mensaje final
    print(f"Todos los datos del usuario '{user_name}' han sido eliminados correctamente.")


def menu_principal():
    while True:
        print("\n--- Menú Principal ---")
        print("1. Capturar imágenes de usuario")
        print("2. Procesar rostros")
        print("3. Generar Embeddings por Usuario")
        print("4. Entrenar Modelo")
        print("5. Registrar Entrada")
        print("6. Registrar Salida")
        print("7. Eliminar Usuario")
        print("8. Salir")

        opcion = input("Selecciona una opción: ")

        if opcion == '1':
            capturar_imagenes()
        elif opcion == '2':
            procesar_rostros()
        elif opcion == '3':
            generar_embeddings_usuario()
        elif opcion == '4':
            entrenar_modelo()
        elif opcion == '5':
            registrar_acceso("Entrada")
        elif opcion == '6':
            registrar_acceso("Salida")
        elif opcion == '7':
            eliminar_usuario()
        elif opcion == '8':
            print("Saliendo del programa...")
            break
        else:
            print("Opción no válida. Intenta de nuevo.")

if __name__ == "__main__":
    menu_principal()
