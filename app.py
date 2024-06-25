from flask import Flask, render_template, request
import joblib
import pandas as pd
import logging
import os

app = Flask(__name__)
# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Definir la ruta de los modelos
MODEL_PATH = 'modelo.pkl'
SCALER_PATH = 'scaler.pkl'

# Verificar la existencia de los archivos de modelo y escalador
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    logging.error(f'No se encontraron archivos de modelo ({MODEL_PATH}) o escalador ({SCALER_PATH}).')

    # Puedes añadir un manejo adicional de error aquí, como lanzar una excepción o cerrar la aplicación
    # Dependiendo de cómo quieras manejar esta situación.

else:
    # Cargar el modelo y el scaler
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        logging.info('Modelo y escalador cargados exitosamente.')
    except Exception as e:
        logging.error(f'Error al cargar el modelo o el escalador: {str(e)}')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'idh' in request.form and 'tma' in request.form and 'vih' in request.form and 'prevalencia' in request.form:
        try:
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
        
        except Exception as e:
            logging.error(f'Error en la predicción: {str(e)}')
            # Puedes manejar el error de otra manera, como redirigir a una página de error o mostrar un mensaje específico al usuario.

    else:
        logging.error('Datos incompletos en la solicitud POST.')

    # En caso de error, redirigir a la página principal
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
