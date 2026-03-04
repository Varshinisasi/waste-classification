import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
import io
from PIL import Image

app = Flask(__name__)

model = tf.keras.models.load_model("waste_classifier.keras")

class_names = ["Hazardous","Non-Recyclable","Organic","Recyclable"]

@app.route("/")
def home():
    return "Waste Classification API Running"

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]
    img = Image.open(file.stream).resize((224,224))

    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    result = class_names[np.argmax(prediction)]

    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)