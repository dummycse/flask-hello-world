import tensorflow_hub as hub
import tensorflow as tf

# Define the path where you want to save the model
local_model_path = 'static/models/arbitrary-image-stylization-v1-256'

# Download and save the model
model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
tf.saved_model.save(model, local_model_path)
