from flask import Flask, render_template, request
# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
#from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import ResNet50

import pickle
from keras.preprocessing import image
import numpy as np
import numpy as np
import cv2
import os
import random
import sqlite3
import uuid
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential, load_model
from PIL import Image
from tensorflow.keras.models import save_model, load_model
from keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/main')
def main():
    return render_template('main.html')
@app.route('/predict')
def predict():
    return render_template('predict.html')
@app.route('/predict',methods=['POST'])
def predictDisease():
    imagefile= request.files['imagefile']
    image_path = "./static/images/" + imagefile.filename
    imagefile.save(image_path)
    # loaded_model = load_model('rice_leaf_model.h5')
    loaded_model = load_model('rice_leaf_model.h5', compile=False)

    #new_image_path = '/content/drive/MyDrive/Test_Cat_Dog/cats/cat.4686.jpg'
    # img = load_img(image_path, target_size=(224, 224))
    # img = img_to_array(img)
    # img_array = np.expand_dims(img_array, axis=0)
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    predictions = loaded_model.predict(img_array)

    # Get the class index with the highest probability
    predicted_class_index = np.argmax(predictions, axis=1)[0]

    # Map the class index to class name
    class_names = ['bacterial_leaf_blight', 'brown_spot', 'healthy', 'leaf_blast', 'leaf_scald', 'narrow_brown_spot']
    predicted_class_name = class_names[predicted_class_index]

    print("Predicted class:", predicted_class_name)


    return render_template('predict.html',image_path=image_path,prediction=predicted_class_name)

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')
@app.route('/charts')
def charts():
    return render_template('charts.html')
@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == "__main__":
    app.run(debug=True)