import cv2
from tensorflow.keras.models import Model, load_model
import numpy as np
from flask import Flask, render_template, request
import imageio

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
    imgData = request.files['img']
    imgData.save('rrr.png')

    try:
        convertImage(imgData)
    except:
        f = request.files['img']
        f.save('image.png')

    labels = {0: 'COVID19', 1: 'NORMAL', 2: 'PNEUMONIA', 3: 'TURBERCULOSIS'}

    img_width, img_height = 224, 224
    image = cv2.imread('rrr.png')
    image = cv2.resize(image, (img_width, img_height))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image[np.newaxis, ...]
    image = image / 255.

    prediction = model.predict(image)
    prediction = np.squeeze(prediction)

    prediction = np.argmax(prediction)
    output = labels[prediction]
    return str(output)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
    app.run(debug=True)
