from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

app = Flask(__name__)
model = load_model('fruit_classifier_model.h5')
class_names = sorted(os.listdir(
    r'C:\Users\Ravi Teja\Desktop\Fruitopia\Fruit\Fruit-Images-Dataset-master\dataset\fruits-360_100x100\fruits-360\Training'
))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction, filename = None, None
    if request.method == 'POST':
        file = request.files['file']
        filename = file.filename
        filepath = os.path.join('static', filename)
        file.save(filepath)
        img = load_img(filepath, target_size=(100, 100))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0) / 255.0
        pred = model.predict(x)
        predicted_class = class_names[np.argmax(pred)]
        prediction = f"Prediction: {predicted_class} (Confidence: {np.max(pred):.2f})"
    return render_template('index.html', prediction=prediction, filename=filename)

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)
