from flask import Flask, request, render_template
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load the trained model
model = load_model('model_vgg16.h5')

# Define the upload folder
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define class labels (adjust based on your training set)
class_labels = ['Bacterialblight','Blast','Brownspot','Tungro']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    # Save the uploaded file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Load and preprocess the image
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)

    # Get predicted label
    predicted_label = class_labels[predicted_class[0]]

    return render_template('index.html', prediction_text=f'Predicted Disease: {predicted_label}', image_path=filepath)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5005,debug=True)
