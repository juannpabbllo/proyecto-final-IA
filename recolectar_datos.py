import cv2
import mediapipe as mp
import csv
import os

# 1. Configuración de MediaPipe (El detector de manos)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# Ajustamos la confianza para que sea preciso
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# 2. Preparamos el archivo CSV (Donde guardaremos los datos)
archivo_csv = 'datos_vocales.csv'
# Si el archivo no existe, creamos los encabezados
if not os.path.exists(archivo_csv):
    with open(archivo_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Creamos 42 columnas (x, y para cada uno de los 21 puntos) + la etiqueta (letra)
        encabezados = ['label']
        for i in range(21):
            encabezados.append(f'x_{i}')
            encabezados.append(f'y_{i}')
        writer.writerow(encabezados)

# 3. Abrimos la cámara
cap = cv2.VideoCapture(0)

print("-------------------------------------------------------")
print("INSTRUCCIONES:")
print("1. Pon tu mano frente a la cámara haciendo una vocal.")
print("2. Presiona la TECLA de la vocal (a, e, i, o, u) para guardar datos.")
print("3. Presiona 'q' para salir.")
print("-------------------------------------------------------")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Invertimos la imagen (efecto espejo) y convertimos a RGB para MediaPipe
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Procesamos la imagen para buscar manos
    results = hands.process(rgb_frame)
    
    # Si encontramos manos...
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dibujamos el esqueleto sobre la mano
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extraemos las coordenadas
            coordenadas = []
            for lm in hand_landmarks.landmark:
                coordenadas.append(lm.x)
                coordenadas.append(lm.y)
            
            # --- CAPTURA DE TECLAS ---
            key = cv2.waitKey(1) & 0xFF
            
            # Si presionas una tecla válida, guardamos
            letra = None
            if key == ord('a'): letra = 0  # Usamos números para entrenar mejor
            elif key == ord('e'): letra = 1
            elif key == ord('i'): letra = 2
            elif key == ord('o'): letra = 3
            elif key == ord('u'): letra = 4
            
            if letra is not None:
                # Guardamos en el CSV
                with open(archivo_csv, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([letra] + coordenadas)
                print(f"Dato guardado para la vocal: {chr(key).upper()}")

    cv2.imshow('Recolector de Vocales', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()