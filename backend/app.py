from flask import Flask, request, jsonify
from flask_cors import CORS



import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from PIL import Image


app = Flask(__name__)


CORS(app, resources={r"/*": {"origins": "*"}})



@app.route('/')
def home():
    return "Rice Leaf Disease API is running"

import tflite_runtime.interpreter as tflite

class_names = [
    'Brown_spot',
    'Bacterial_Blight',
    'Healthy',
    'Leaf_Blast'
]


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['file']

        img = Image.open(file).convert("RGB")
        img = img.resize((224, 224))

        img = np.array(img, dtype=np.float32)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        interpreter.set_tensor(
            input_details[0]['index'],
            img
        )

        interpreter.invoke()

        prediction = interpreter.get_tensor(
            output_details[0]['index']
        )

        result = class_names[int(np.argmax(prediction))]

        return jsonify({
            "prediction": result
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({
            "error": str(e)
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)