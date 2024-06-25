from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)
# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo y el scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Obtener datos del formulario
    idh = float(request.form['idh'])
    tma = float(request.form['tma'])
    vih = float(request.form['vih'])
    prevalencia = float(request.form['prevalencia'])

    # Crear un DataFrame con los datos
    data = pd.DataFrame({
        'Income composition of resources': [idh],
        'Adult Mortality': [tma],
        ' HIV/AIDS': [vih],
        ' thinness  1-19 years': [prevalencia]
    })

    # Escalar los datos
    data_scaled = scaler.transform(data)

    # Realizar la predicción
    prediction = model.predict(data_scaled)[0]

    # Mostrar el resultado al usuario
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
