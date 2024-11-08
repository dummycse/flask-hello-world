from flask import Flask, render_template, request, redirect, url_for
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import os
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

local_model_path = 'static/models/arbitrary-image-stylization-v1-256'
model = tf.saved_model.load(local_model_path)

# Load the TensorFlow Hub model once at startup
# model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Function to load and preprocess images
def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling image uploads and processing
@app.route('/generate', methods=['POST'])
def generate():
    content_img = request.files['content_image']
    style_img = request.files['style_image']
    
    content_path = os.path.join(app.config['UPLOAD_FOLDER'], 'content.jpg')
    style_path = os.path.join(app.config['UPLOAD_FOLDER'], 'style.jpg')
    stylized_path = os.path.join(app.config['UPLOAD_FOLDER'], 'stylized.jpg')
    
    content_img.save(content_path)
    style_img.save(style_path)

    # Load images and apply style transfer
    content_image = load_image(content_path)
    style_image = load_image(style_path)
    stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]

    # Convert tensor to image and save
    stylized_image_np = np.squeeze(stylized_image.numpy())
    stylized_image_np = (stylized_image_np * 255).astype(np.uint8)
    Image.fromarray(stylized_image_np).save(stylized_path)

    return redirect(url_for('result'))

# Route to display the result
@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
