import cv2
import mediapipe as mp
import pickle
import numpy as np

# 1. Cargar el modelo entrenado (El Cerebro)
try:
    with open('modelo_vocales.p', 'rb') as f:
        modelo = pickle.load(f)
except FileNotFoundError:
    print("¡Error! No encuentro el archivo 'modelo_vocales.p'.")
    print("Asegúrate de haber ejecutado 'entrenar_modelo.py' primero.")
    exit()

# 2. Configuración de MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, 
                       max_num_hands=1, 
                       min_detection_confidence=0.5, 
                       min_tracking_confidence=0.5)

# Diccionario para traducir números a letras
# Asegúrate de que coincida con lo que entrenaste
etiquetas = {0: 'A', 1: 'E', 2: 'I', 3: 'O', 4: 'U'}

# 3. Encender la cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Espejo y corrección de color
    frame = cv2.flip(frame, 1) # Espejo
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar la imagen
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            # --- PREPARAR LOS DATOS IGUAL QUE EN EL ENTRENAMIENTO ---
            data_aux = []
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x)
                data_aux.append(lm.y)
            
            # --- PREDICCIÓN ---
            # El modelo espera una lista de listas, por eso los corchetes extra [data_aux]
            prediccion = modelo.predict([data_aux])
            letra_detectada = etiquetas[int(prediccion[0])]

            # --- DIBUJAR EN PANTALLA ---
            # Dibujar el esqueleto de la mano
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Dibujar un recuadro y la letra
            h, w, c = frame.shape
            cv2.rectangle(frame, (0,0), (180, 80), (0,0,0), -1) # Fondo negro arriba izq
            cv2.putText(frame, f"Letra: {letra_detectada}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow('Detector de Vocales ASL', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()