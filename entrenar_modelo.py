import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# 1. Cargar los datos
print("Cargando tus 3100 datos...")
try:
    data = pd.read_csv('datos_vocales.csv')
except FileNotFoundError:
    print("¡ERROR! No encontré el archivo 'datos_vocales.csv'.")
    exit()

# 2. Separar las preguntas (coordenadas) de las respuestas (letras)
X = data.drop('label', axis=1) # Las coordenadas
y = data['label']              # La etiqueta correcta

# 3. Dividir en Entrenamiento y Examen
# Usamos el 20% de los datos para hacerle un examen final al modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Entrenar el Modelo
print("Entrenando el cerebro digital...")
modelo = RandomForestClassifier(n_estimators=100)
modelo.fit(X_train, y_train)

# 5. Evaluar precisión
y_pred = modelo.predict(X_test)
precision = accuracy_score(y_test, y_pred)

print(f"--------------------------------")
print(f"¡ENTRENAMIENTO COMPLETADO!")
print(f"Precisión del modelo: {precision * 100:.2f}%")
print(f"--------------------------------")

# 6. Guardar el modelo
with open('modelo_vocales.p', 'wb') as f:
    pickle.dump(modelo, f)
print("Modelo guardado exitosamente como 'modelo_vocales.p'")