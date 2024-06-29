from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
MODEL_PATH = 'model/cnn_model.h5'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


try:
    model = load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except OSError as e:
    print(f"Error loading model: {e}")
    model = None


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    return img_array


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':

        file = request.files['file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        if model:
            img = preprocess_image(filepath)
            prediction = model.predict(img)

            disease = np.argmax(prediction, axis=1)

            return render_template('index.html', prediction=disease, img_path=filepath)
        else:
            return render_template('index.html', prediction="Model not loaded", img_path=None)

    return render_template('index.html', prediction=None, img_path=None)


if __name__ == '__main__':

    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    app.run(debug=True)
