import cv2
from tensorflow.keras.models import Model, load_model
import numpy as np
from flask import Flask, render_template, request
import pyrebase


config = {
  'apiKey': "AIzaSyCvQScWOdmOAh7sjh-fbOtfRPs-wmOF640",
  'authDomain': "xraydiagnos.firebaseapp.com",
  'projectId': "xraydiagnos",
  'storageBucket': "xraydiagnos.appspot.com",
  'messagingSenderId': "371080730510",
  'appId': "1:371080730510:web:a8c2e0587e3a4f3a2ef6df",
    'databaseURL': "https://xraydiagnos.firebaseapp.com"
}

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
    imgData = request.files['img']
    imgData.save(f'{fioData}.png')

    try:
        convertImage(imgData)
    except:
        f = request.files['img']
        f.save('image.png')

    labels = {0: 'COVID19', 1: 'NORMAL', 2: 'PNEUMONIA', 3: 'TURBERCULOSIS'}

    img_width, img_height = 224, 224
    image = cv2.imread(f'{fioData}.png')
    image = cv2.resize(image, (img_width, img_height))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image[np.newaxis, ...]
    image = image / 255.

    prediction = model.predict(image)
    prediction = np.squeeze(prediction)

    prediction = np.argmax(prediction)
    output = labels[prediction]
#------------------------------------------------------------------------
    firebase = pyrebase.initialize_app(config)
    storage = firebase.storage()

#дать на выпуск fioata, а загружать rrr
    path_local = f"{fioData}.png"

    storage.child(path_local).put(path_local)
# ------------------------------------------------------------------------
    return str(output)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
    app.run(debug=True)
