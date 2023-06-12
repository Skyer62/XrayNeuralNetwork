import cv2
from keras.models import Model, load_model
import numpy as np
from flask import Flask, render_template, request
import pyrebase
from transliterate import translit, get_available_language_codes
from datetime import datetime

image_size = 299
batch_size = 32

model = load_model("inceptionv3_fine_tuned.h5")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

import re
import base64

def convertImage(imgData1):
    imgstr = re.search(r'base64,(.*)', str(imgData1)).group(1)
    with open('output.png', 'wb') as output:
        output.write(base64.b64decode(imgstr))

@app.route('/predict/', methods=['POST'])
def predict():
    fioData = request.form['fio']
    fioData.replace(" ", "")
    fioData = translit(fioData, language_code='ru', reversed=True)

    imgData = request.files['img']
    imgData.save('rrr.png')

    try:
        convertImage(imgData)
    except:
        f = request.files['img']
        f.save('image.png')

    labels = {0: 'COVID-19', 1: 'Норма', 2: 'Пневмония', 3: 'Туберкулёз'}

    img_width, img_height = 224, 224
    image = cv2.imread(f'rrr.png')
    image = cv2.imread('rrr.png')
    image = cv2.resize(image, (img_width, img_height))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image[np.newaxis, ...]
    image = image / 255.

    prediction = model.predict(image)
    prediction = np.squeeze(prediction)

    prediction = np.argmax(prediction)
    output = labels[prediction]

    config = {
        'apiKey': "AIzaSyCvQScWOdmOAh7sjh-fbOtfRPs-wmOF640",
        'authDomain': "xraydiagnos.firebaseapp.com",
        'projectId': "xraydiagnos",
        'storageBucket': "xraydiagnos.appspot.com",
        'messagingSenderId': "371080730510",
        'appId': "1:371080730510:web:a8c2e0587e3a4f3a2ef6df",
        'databaseURL': "https://xraydiagnos.firebaseapp.com"
    }

    firebase = pyrebase.initialize_app(config)
    storage = firebase.storage()

    path_local = "rrr.png"
    path_load = f"images/{fioData + datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.png"

    storage.child(path_load).put(path_local)

    return str(output)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
    app.run(debug=True)